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
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
testset = torchvision.datasets.CIFAR10(
    root='../../cifar10/resnet32/data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
loader_small = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

                 
model = ResNet32(num_classes=NUM_CLASSES, defense=False)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval().to(device)

                         
TARGET_CLASS = 2
PATCH_Y, PATCH_X, PATCH_H, PATCH_W = 21, 21, 10, 10
SIGMA = 1e-4
EOT_M = 20
os.makedirs('./result', exist_ok=True)
criterion = nn.CrossEntropyLoss()

                                                    
sni = np.array([TARGET_CLASS])
np.save('./result/SNI.npy', sni)


                                        
def optimize_trigger_eot(model, loader, n_steps=100, lr=0.01):
    trigger = torch.zeros(3, PATCH_H, PATCH_W, device=device)
    for step in range(n_steps):
        trigger.requires_grad_(True)
        total_grad = torch.zeros_like(trigger)
        for x, _ in loader:
            x = x.to(device).clone()
            x[:, :, PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W] = trigger
            y = torch.full((x.size(0),), TARGET_CLASS, dtype=torch.long, device=device)
            for _ in range(EOT_M):
                saved = {n: p.data.clone() for n, p in model.named_parameters()}
                for p in model.parameters():
                    p.data.add_(torch.randn_like(p) * SIGMA)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if trigger.grad is not None:
                    total_grad.add_(trigger.grad.data)
                    trigger.grad.zero_()
                for n, p in model.named_parameters():
                    p.data.copy_(saved[n])
            break
        with torch.no_grad():
            trigger = (trigger - lr * (total_grad / EOT_M).sign()).clamp(-0.5, 0.5)
        trigger = trigger.detach()
        if (step + 1) % 20 == 0:
            print(f"  EOT Trigger step {step + 1}/{n_steps}")
    return trigger.detach()


trigger = optimize_trigger_eot(model, loader_small)
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
