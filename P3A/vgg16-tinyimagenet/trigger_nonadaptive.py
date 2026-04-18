import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

                        
MODEL_DIR = '../../tinyimagenet/vgg16'
sys.path.insert(0, MODEL_DIR)
from VGG import VGG16_Tiny

NUM_CLASSES = 200
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
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

          
sni = np.array([TARGET_CLASS])
np.save('./result/SNI.npy', sni)


                                    
def optimize_trigger(model, loader, n_steps=100, lr=0.01):
    trigger = torch.zeros(3, PATCH_H, PATCH_W, device=device)
    for step in range(n_steps):
        trigger.requires_grad_(True)
        for x, _ in loader:
            x = x.to(device).clone()
            x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = trigger
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            loss = criterion(model(x), y)
            loss.backward()
            break
        with torch.no_grad():
            trigger = (trigger - lr * trigger.grad.sign()).clamp(-0.5, 0.5)
        trigger = trigger.detach()
        if (step + 1) % 20 == 0:
            print(f"  Trigger step {step + 1}/{n_steps}, loss={loss.item():.4f}")
    return trigger.detach()


trigger = optimize_trigger(model, loader_small)
torch.save(trigger, './result/perturbed.pth')
print(f"Trigger saved, shape={trigger.shape}")

                                
model.eval()
c = t = 0
with torch.no_grad():
    for x, _ in loader_test:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = trigger
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
print(f"Trigger ASR: {100. * c / t:.2f}%")
