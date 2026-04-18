import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

                                                                 
              
                                                                 
MODEL_DIR = '../../tinyimagenet/vgg16'
sys.path.insert(0, MODEL_DIR)
from VGG import VGG16_Tiny

NUM_CLASSES = 200
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                                                 
      
                                                                 
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_dir = os.path.join(MODEL_DIR, 'tiny-imagenet-200', 'val')
testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
loader_test  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32,  shuffle=False, num_workers=2)

                                                                 
       
                                                                 
model = VGG16_Tiny(num_classes=NUM_CLASSES, defense=False)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)

                                                                 
               
                                                                 
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
SIGMA = 1e-4
EOT_M = 20
os.makedirs('./result', exist_ok=True)
trigger = torch.load('./result/perturbed.pth', map_location=device)                   
criterion = nn.CrossEntropyLoss()

                                                                 
                      
                                                                 
def get_quant_params(w_flat):
    w_min = w_flat.min().item()
    w_max = w_flat.max().item()
    if w_max == w_min:
        return None, None
    return w_min, (w_max - w_min) / 255.0

def count_bit_flips(a, b):
    x = int(a) ^ int(b)
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

                                                                 
                     
                                                                 
def eot_grad_weights(model, x, y_target, M, sigma, criterion):
    model.zero_grad()
    for _ in range(M):
        saved = {n: p.data.clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.data.add_(torch.randn_like(p) * sigma)
        logits = model(x)
        loss = criterion(logits, y_target)
        loss.backward()
        for n, p in model.named_parameters():
            p.data.copy_(saved[n])
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(M)

                                                                 
                    
                                                                 
def eval_clean(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def eval_asr(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(device)
            x[:, :, PATCH_Y:PATCH_Y+PATCH_H, PATCH_X:PATCH_X+PATCH_W] = trigger
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100. * correct / total

                                                                 
                                             
                                                                 
def find_sensitive_layer(model):
    model.eval()
    for x, _ in loader_small:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y+PATCH_H, PATCH_X:PATCH_X+PATCH_W] = trigger
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        eot_grad_weights(model, x, y, EOT_M, SIGMA, criterion)
        break
    scores = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
            g = m.weight.grad.data.flatten()
            w = m.weight.data.flatten()
            wmin, delta = get_quant_params(w)
            if delta is None:
                scores.append(0.)
                continue
            idx = ((w - wmin) / delta).round().clamp(0, 255).long()
            best = 0.
            for bit in range(8):
                fi = (idx ^ (1 << bit)).clamp(0, 255)
                nw = wmin + fi.float() * delta
                s = (g * (nw - w)).abs().max().item()
                best = max(best, s)
            scores.append(best)
        else:
            scores.append(0.)
    return int(np.argmax(scores)) + 1


def find_vulnerable_bit(model, psens):
    model.eval()
    for x, _ in loader_small:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y+PATCH_H, PATCH_X:PATCH_X+PATCH_W] = trigger
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        eot_grad_weights(model, x, y, EOT_M, SIGMA, criterion)
        break
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n += 1
            if n == psens and m.weight.grad is not None:
                g = m.weight.grad.data.flatten()
                w = m.weight.data.flatten()
                wmin, delta = get_quant_params(w)
                if delta is None:
                    return 0, 0
                idx = ((w - wmin) / delta).round().clamp(0, 255).long()
                be, bb, bs = 0, 0, 0.
                for i in range(len(g)):
                    for bit in range(8):
                        fi = max(0, min(255, idx[i].item() ^ (1 << bit)))
                        nw = wmin + fi * delta
                        s = abs(g[i].item()) * abs(nw - w[i].item())
                        if s > bs:
                            bs, be, bb = s, i, bit
                return be, bb
    return 0, 0


def apply_bit_flip(model, psens, elem_loc, bit):
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n += 1
            if n == psens:
                w = m.weight.data.flatten().clone()
                wmin, delta = get_quant_params(w)
                if delta is None:
                    return w[elem_loc].item(), w[elem_loc].item(), 0
                idx = ((w - wmin) / delta).round().clamp(0, 255).long()
                oi = idx[elem_loc].item()
                ni = max(0, min(255, oi ^ (1 << bit)))
                ov = w[elem_loc].item()
                nv = wmin + ni * delta
                w[elem_loc] = nv
                m.weight.data = w.reshape(m.weight.data.shape)
                return ov, nv, count_bit_flips(oi, ni)
    return None, None, 0

                                                                 
             
                                                                 
B_f = 100
psens = find_sensitive_layer(model)
print(f"Sensitive layer: {psens}")

n_b, last_elem, num_iter = 0, -1, 0
x_axis, y_axis = [], []
fig = plt.figure(figsize=(10, 6))

while n_b < B_f:
    elem_loc, bit = find_vulnerable_bit(model, psens)
    if elem_loc == last_elem:
        num_iter += 1
    if num_iter >= 8:
        num_iter = 0
    last_elem = elem_loc

    ov, nv, flips = apply_bit_flip(model, psens, elem_loc, bit)
    if ov is None:
        break
    n_b += flips

    asr   = eval_asr(model)
    clean = eval_clean(model)
    print(f"Bit flips: {n_b}  ASR: {asr:.2f}%  Clean acc: {clean:.2f}%")
    x_axis.append(n_b)
    y_axis.append(asr)
    plt.clf()
    plt.plot(x_axis, y_axis)
    plt.xlabel('Bit flips')
    plt.ylabel('ASR (%)')
    fig.savefig('./result/asr_proflip_adaptive.png', bbox_inches='tight')

    if asr >= 90.0:
        print("Reached target ASR")
        break

print(f"Final: {n_b} bit flips, ASR={eval_asr(model):.2f}%")
