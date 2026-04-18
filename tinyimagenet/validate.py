import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset_tinyimagenet import get_tinyimagenet_loaders


def compute_tau(ari_model, val_loader, device):
    margins = []
    ari_model.eval()
    base = ari_model.model
    sigma = ari_model.sigma
    M = ari_model.M_small
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
                                         
            logits_list = []
            for _ in range(M):
                noisy = {n: p + torch.randn_like(p) * sigma
                         for n, p in base.named_parameters()}
                logits_list.append(base.forward_with_params(x, noisy))
            avg_logits = torch.stack(logits_list).mean(0)
            probs = F.softmax(avg_logits, dim=1)
            p_sorted, _ = probs.sort(dim=1, descending=True)
            m = (p_sorted[:, 0] - p_sorted[:, 1]).cpu().tolist()
            margins.extend(m)
    margins = torch.tensor(margins)
    return (margins.mean() - 1.28 * margins.std()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet32', choices=['resnet32', 'vgg16'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--key', type=int, default=2025)
    parser.add_argument('--sigma', type=float, default=1e-4)
    parser.add_argument('--M_small', type=int, default=5)
    parser.add_argument('--M_large', type=int, default=25)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 200
    _, val_loader = get_tinyimagenet_loaders()

    if args.arch == 'resnet32':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'resnet32'))
        from ResNet import ResNet32_Tiny, ARIWrapper
        base_model = ResNet32_Tiny(num_classes=num_classes, defense=False)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vgg16'))
        from VGG import VGG16_Tiny, ARIWrapper
        base_model = VGG16_Tiny(num_classes=num_classes, defense=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    base_model.load_state_dict(ckpt)
    base_model.eval().to(device)

                                     
    ari_model = ARIWrapper(base_model, sigma=args.sigma, M_small=args.M_small,
                           M_large=args.M_large, tau=0.0)
    ari_model.to(device)

    print("Computing tau from clean validation data...")
    tau = compute_tau(ari_model, val_loader, device)
    print(f"tau = {tau:.4f}")
    ari_model.tau = tau

              
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0, 0, 0
    ari_model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = ari_model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Val loss: {total_loss/total:.4f}  Val acc: {100.*total_correct/total:.2f}%")


if __name__ == '__main__':
    main()
