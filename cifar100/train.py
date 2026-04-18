import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset_cifar100 import get_cifar100_loaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet32', choices=['resnet32', 'vgg16'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--key', type=int, default=2025)
    parser.add_argument('--bits', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 100
    train_loader, val_loader = get_cifar100_loaders(batch_size=args.batch_size)

                                                                      
    if args.arch == 'resnet32':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'resnet32'))
        from ResNet import ResNet32, apply_kcr_resnet32, quantize_affine
        model = ResNet32(num_classes=num_classes, defense=False)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vgg16'))
        from VGG import VGG16, apply_kcr_vgg16, quantize_affine
        model = VGG16(num_classes=num_classes, defense=False)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    ckpt_dir = os.path.join(os.path.dirname(__file__), args.arch, 'checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}: train_acc={tr_acc*100:.2f}%  val_acc={val_acc*100:.2f}%")
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'clean_best.pth'))

                      
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'clean_final.pth'))
    print(f"Best val acc: {best_acc*100:.2f}%")

                                                    
    with torch.no_grad():
        if args.arch == 'resnet32':
            apply_kcr_resnet32(model, key=args.key)
        else:
            apply_kcr_vgg16(model, key=args.key)
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                param.data = quantize_affine(param.data, bits=args.bits)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'defended.pth'))
    print(f"Defended model saved to {ckpt_dir}/defended.pth")


if __name__ == '__main__':
    main()
