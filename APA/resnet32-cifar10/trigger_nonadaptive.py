import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

                        
MODEL_DIR = '../../cifar10/resnet32'
sys.path.insert(0, MODEL_DIR)
from ResNet import ResNet32

NUM_CLASSES = 10
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
testset = torchvision.datasets.CIFAR10(
    root='../../cifar10/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

                 
model = ResNet32(num_classes=NUM_CLASSES, defense=False)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval().to(device)

                         
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
os.makedirs('./result', exist_ok=True)
criterion = nn.CrossEntropyLoss()

                                                                               
sni = np.array([TARGET_CLASS])
np.save('./result/SNI.npy', sni)
print(f"SNI: {sni}")

                                           
EPSILON = 8. / 255.
N_STEPS = 100
STEP_SIZE = EPSILON * 2.5 / N_STEPS

patch = torch.zeros(3, PATCH_H, PATCH_W, device=device)

for step in range(N_STEPS):
    patch.requires_grad_(True)
    for x, _ in loader_small:
        x = x.to(device).clone()
        x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = patch
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        loss = criterion(model(x), y)
        loss.backward()
        break
    with torch.no_grad():
        patch = (patch + STEP_SIZE * patch.grad.sign()).clamp(-EPSILON, EPSILON)
    patch = patch.detach()
    if (step + 1) % 20 == 0:
        print(f"  Step {step + 1}/{N_STEPS}")

torch.save(patch, './result/perturbed.pth')
print(f"Trigger saved: shape={patch.shape}")

                        
model.eval()
c = t = 0
with torch.no_grad():
    for x, _ in loader_test:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = patch
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
print(f"Trigger ASR (nonadaptive): {100. * c / t:.2f}%")
