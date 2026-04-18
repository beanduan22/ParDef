import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

                                                                 
              
                                                                 
MODEL_DIR = '../../cifar10/vgg16'
sys.path.insert(0, MODEL_DIR)
from VGG import VGG16

NUM_CLASSES = 10
CKPT = os.path.join(MODEL_DIR, 'checkpoint', 'defended.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                                                 
      
                                                                 
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

testset = torchvision.datasets.CIFAR10(
    root=os.path.join(MODEL_DIR, 'data'),
    train=False,
    download=False,
    transform=transform_test,
)
loader_test  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32,  shuffle=False, num_workers=2)

                                                                 
       
                                                                 
model = VGG16(num_classes=NUM_CLASSES, defense=False)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)

                                                                 
               
                                                                 
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
os.makedirs('./result', exist_ok=True)
criterion = nn.CrossEntropyLoss()

                                                                 
                                               
                                                                 
def compute_sni(model, loader, target_class, device):
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        break
    sni = np.array([target_class])
    return sni

sni = compute_sni(model, loader_small, TARGET_CLASS, device)
np.save('./result/SNI.npy', sni)
print(f"SNI saved: {sni}")

                                                                 
                                   
                                                                 
def optimize_trigger(model, loader, target_class, patch_h, patch_w, n_steps=100, lr=0.01):
    trigger = torch.zeros(3, patch_h, patch_w, device=device)

    for step in range(n_steps):
        trigger.requires_grad_(True)
        total_loss = torch.tensor(0., device=device)
        for x, _ in loader:
            x = x.to(device).clone()
            x[:, :, PATCH_Y:PATCH_Y+PATCH_H, PATCH_X:PATCH_X+PATCH_W] = trigger
            y = torch.full((x.size(0),), target_class, dtype=torch.long, device=device)
            logits = model(x)
            total_loss = criterion(logits, y)
            break
        total_loss.backward()
        with torch.no_grad():
            trigger = trigger - lr * trigger.grad.sign()
            trigger = trigger.clamp(-0.5, 0.5)
        trigger = trigger.detach()
        if (step + 1) % 10 == 0:
            print(f"  Trigger step {step+1}/{n_steps}, loss={total_loss.item():.4f}")
    return trigger.detach()

trigger = optimize_trigger(model, loader_small, TARGET_CLASS, PATCH_H, PATCH_W, n_steps=100, lr=0.01)
torch.save(trigger, './result/perturbed.pth')
print(f"Trigger saved to ./result/perturbed.pth, shape={trigger.shape}")

                                                                 
                      
                                                                 
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in loader_test:
        x = x.to(device)
        x[:, :, PATCH_Y:PATCH_Y+PATCH_H, PATCH_X:PATCH_X+PATCH_W] = trigger
        y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
print(f"Trigger ASR (nonadaptive): {100.*correct/total:.2f}%")
