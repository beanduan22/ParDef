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

                        
MODEL_DIR = '../../tinyimagenet/vgg16'
sys.path.insert(0, MODEL_DIR)
from VGG import VGG16_Tiny

NUM_CLASSES = 200
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
testset = torchvision.datasets.ImageFolder(
    '../../tinyimagenet/vgg16/tiny-imagenet-200/val', transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

                 
model = VGG16_Tiny(num_classes=NUM_CLASSES, defense=False)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval().to(device)

                         
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
os.makedirs('./result', exist_ok=True)
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


def eval_asr(m, patch):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for x, _ in loader_test:
            x = x.to(device)
            x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = patch
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            c += (m(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return 100. * c / t


                            
def pgd_patch(model, loader, epsilon, n_steps=50, step_size=None):
    if step_size is None:
        step_size = epsilon * 2.5 / n_steps

    patch = torch.zeros(3, PATCH_H, PATCH_W, device=device)

    for step in range(n_steps):
        patch.requires_grad_(True)
        total_loss = 0.
        n_batches = 0
        for x, _ in loader:
            x = x.to(device).clone()
            x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = patch
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss
            n_batches += 1
            if n_batches >= 2:
                break
        (total_loss / n_batches).backward()
        with torch.no_grad():
            patch = (patch + step_size * patch.grad.sign()).clamp(-epsilon, epsilon)
        patch = patch.detach()
    return patch.detach()


                                           
print(f"Clean accuracy: {eval_clean(model):.2f}%")

EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]
x_axis, y_axis = [], []
fig = plt.figure(figsize=(10, 6))

for eps in EPSILONS:
    patch = pgd_patch(model, loader_small, epsilon=eps, n_steps=50)
    asr = eval_asr(model, patch)
    print(f"eps={eps * 255:.0f}/255  ASR: {asr:.2f}%")

    torch.save(patch, f'./result/patch_eps{int(eps * 255)}.pth')

    x_axis.append(eps * 255)
    y_axis.append(asr)
    plt.clf()
    plt.plot(x_axis, y_axis, 'o-')
    plt.xlabel('epsilon (x1/255)')
    plt.ylabel('ASR (%)')
    fig.savefig('./result/asr_apa.png', bbox_inches='tight')

                                                       
best_eps = 8 / 255
patch_default = pgd_patch(model, loader_small, epsilon=best_eps, n_steps=100)
torch.save(patch_default, './result/perturbed.pth')
print(f"Default trigger saved (eps=8/255): ASR={eval_asr(model, patch_default):.2f}%")
