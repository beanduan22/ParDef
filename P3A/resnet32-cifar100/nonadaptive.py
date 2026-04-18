import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

                        
MODEL_DIR = '../../cifar100/resnet32'
sys.path.insert(0, MODEL_DIR)
from ResNet import ResNet32

NUM_CLASSES = 100
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
testset = torchvision.datasets.CIFAR100(
    root='../../cifar100/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

                 
model = ResNet32(num_classes=NUM_CLASSES, defense=False)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval().to(device)

                         
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
os.makedirs('./result', exist_ok=True)
trigger = torch.load('./result/perturbed.pth', map_location=device)
criterion = nn.CrossEntropyLoss()


                      
def eval_clean(m):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader_test:
            x, y = x.to(device), y.to(device)
            c += (m(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return 100. * c / t


def eval_asr(m):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for x, _ in loader_test:
            x = x.to(device)
            x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = trigger
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            c += (m(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return 100. * c / t


                    
def compute_channel_gradients(model):
    model.eval()
    model.zero_grad()
    for x, _ in loader_small:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = trigger
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        loss = criterion(model(x), y)
        loss.backward()
        break

    layer_channel_grads = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
            g = m.weight.grad.data
            if g.dim() == 4:
                norms = g.norm(dim=(1, 2, 3))
            else:
                norms = g.norm(dim=1)
            layer_channel_grads.append((m, norms))
    return layer_channel_grads


def perturb_top_k_channels(model, k):
    layer_grads = compute_channel_gradients(model)

    all_channels = []
    for li, (m, norms) in enumerate(layer_grads):
        for ci, norm in enumerate(norms):
            all_channels.append((norm.item(), li, ci))

    all_channels.sort(reverse=True)
    selected = all_channels[:k]

    with torch.no_grad():
        for norm_val, li, ci in selected:
            m, _ = layer_grads[li]
            g = m.weight.grad.data
            w = m.weight.data

            w_flat = w.flatten()
            w_min = w_flat.min().item()
            w_max = w_flat.max().item()
            if w_max == w_min:
                continue
            delta = (w_max - w_min) / 255.0

            if g.dim() == 4:
                perturbation = delta * g[ci].sign()
                w[ci] = w[ci] + perturbation
            else:
                perturbation = delta * g[ci].sign()
                w[ci] = w[ci] + perturbation

            w_new_flat = w.flatten()
            w_new_min = w_new_flat.min().item()
            w_new_max = w_new_flat.max().item()
            if w_new_max > w_new_min:
                new_delta = (w_new_max - w_new_min) / 255.0
                idx = ((w.flatten() - w_new_min) / new_delta).round().clamp(0, 255)
                w.copy_((w_new_min + idx * new_delta).reshape(w.shape))


                            
print(f"Initial: clean={eval_clean(model):.2f}%  ASR={eval_asr(model):.2f}%")

K_VALUES = [1, 3, 5, 7]

x_axis, y_axis = [], []
fig = plt.figure(figsize=(10, 6))

for k in K_VALUES:
    perturb_top_k_channels(model, k)
    asr = eval_asr(model)
    clean = eval_clean(model)
    print(f"Channels perturbed: {k}  ASR: {asr:.2f}%  Clean acc: {clean:.2f}%")
    x_axis.append(k)
    y_axis.append(asr)
    plt.clf()
    plt.plot(x_axis, y_axis)
    plt.xlabel('Channels perturbed')
    plt.ylabel('ASR (%)')
    fig.savefig('./result/asr_p3a.png', bbox_inches='tight')
    if asr >= 90.0:
        print("Reached target ASR")
        break

print(f"Final ASR: {eval_asr(model):.2f}%")
