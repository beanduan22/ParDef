import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_affine(weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << bits) - 1
    w_min = weight.min()
    w_max = weight.max()
    if w_max == w_min:
        return weight.clone()
    delta = (w_max - w_min) / qmax
    q_idx = torch.clamp(((weight - w_min) / delta).round(), 0, qmax)
    return q_idx * delta + w_min


def _xform_weight(w: torch.Tensor,
                  perm_out: torch.Tensor, scale_out: torch.Tensor,
                  perm_in: torch.Tensor,  scale_in: torch.Tensor) -> torch.Tensor:
    if w.dim() == 4:
        W = w[perm_out][:, perm_in].clone()
        W = W * scale_out.view(-1, 1, 1, 1) / scale_in.view(1, -1, 1, 1)
    else:
        W = w[perm_out][:, perm_in].clone()
        W = W * scale_out.view(-1, 1) / scale_in.view(1, -1)
    return W


def _xform_bias(b: torch.Tensor,
                perm_out: torch.Tensor, scale_out: torch.Tensor) -> torch.Tensor:
    return (scale_out * b[perm_out]).clone()


def _update_bn(bn: nn.BatchNorm2d,
               perm_out: torch.Tensor, scale_out: torch.Tensor) -> None:
    with torch.no_grad():
        bn.weight.data       = (scale_out * bn.weight.data[perm_out]).clone()
        bn.bias.data         = (scale_out * bn.bias.data[perm_out]).clone()
        bn.running_mean.data = (scale_out * bn.running_mean.data[perm_out]).clone()
        bn.running_var.data  = (scale_out ** 2 * bn.running_var.data[perm_out]).clone()


def apply_kcr_resnet32(model: 'ResNet32', key: int) -> None:
    rng = torch.Generator()
    rng.manual_seed(key)

    def rperm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=rng)

    def rscale(n: int) -> torch.Tensor:
        return torch.rand(n, generator=rng) * 0.5 + 0.75

    with torch.no_grad():
                                                              
                                           
        conv0 = model.init_conv[0]
        bn0   = model.init_conv[1]
        perm_in  = torch.arange(conv0.weight.shape[1])
        scale_in = torch.ones(conv0.weight.shape[1])
        perm_out  = rperm(conv0.weight.shape[0])
        scale_out = rscale(conv0.weight.shape[0])

        conv0.weight.data = _xform_weight(conv0.weight.data, perm_out, scale_out,
                                          perm_in, scale_in)
        _update_bn(bn0, perm_out, scale_out)

        cur_perm  = perm_out
        cur_scale = scale_out

                                           
        for stage in [model.layer1, model.layer2, model.layer3]:
            for block in stage:
                blk_perm  = cur_perm.clone()
                blk_scale = cur_scale.clone()

                                                        
                c1, bn1  = block.conv1, block.bn1
                perm_c1  = rperm(c1.weight.shape[0])
                scale_c1 = rscale(c1.weight.shape[0])
                c1.weight.data = _xform_weight(c1.weight.data, perm_c1, scale_c1,
                                               blk_perm, blk_scale)
                _update_bn(bn1, perm_c1, scale_c1)

                                          
                c2, bn2         = block.conv2, block.bn2
                has_skip_conv   = len(list(block.shortcut.children())) > 0

                if has_skip_conv:
                                                                                   
                    perm_c2  = rperm(c2.weight.shape[0])
                    scale_c2 = rscale(c2.weight.shape[0])
                    skip_children = list(block.shortcut.children())
                    skip_conv, skip_bn = skip_children[0], skip_children[1]
                    skip_conv.weight.data = _xform_weight(
                        skip_conv.weight.data, perm_c2, scale_c2, blk_perm, blk_scale
                    )
                    _update_bn(skip_bn, perm_c2, scale_c2)
                else:
                                                                                          
                    perm_c2  = blk_perm.clone()
                    scale_c2 = blk_scale.clone()

                c2.weight.data = _xform_weight(c2.weight.data, perm_c2, scale_c2,
                                               perm_c1, scale_c1)
                _update_bn(bn2, perm_c2, scale_c2)

                cur_perm  = perm_c2
                cur_scale = scale_c2

                                              
        fc = model.fc
        perm_fc_out  = torch.arange(fc.weight.shape[0])
        scale_fc_out = torch.ones(fc.weight.shape[0])
        fc.weight.data = _xform_weight(fc.weight.data, perm_fc_out, scale_fc_out,
                                       cur_perm, cur_scale)
        if fc.bias is not None:
            fc.bias.data = _xform_bias(fc.bias.data, perm_fc_out, scale_fc_out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=10, defense=True, key=1234, bits=8):
        super(ResNet32, self).__init__()
        self.in_channels = 16
        self.defense = defense
        self.key = key
        self.bits = bits

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

        if self.defense:
            with torch.no_grad():
                                                                                        
                apply_kcr_resnet32(self, self.key)
                                                                                         
                for name, param in self.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        param.data = quantize_affine(param.data, bits=self.bits)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, channels, s))
            self.in_channels = channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def forward_with_params(self, x, noisy_params):
        backup = {}
        for name, p in self.named_parameters():
            backup[name] = p.data.clone()
            p.data = noisy_params[name].data
        out = self.forward(x)
        for name, p in self.named_parameters():
            p.data = backup[name]
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ARIWrapper(nn.Module):
    def __init__(self, model, sigma=1e-4, M_small=5, M_large=25, tau=0.1):
        super().__init__()
        self.model   = model
        self.sigma   = sigma
        self.M_small = M_small
        self.M_large = M_large
        self.tau     = tau

    def stochastic_forward(self, x: torch.Tensor, M: int) -> torch.Tensor:
        logits_all = []
        for _ in range(M):
            noisy = {n: p + torch.randn_like(p) * self.sigma
                     for n, p in self.model.named_parameters()}
            logits_all.append(self.model.forward_with_params(x, noisy))
        return torch.stack(logits_all, dim=0)              

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

                                                               
        logits_s   = self.stochastic_forward(x, self.M_small)                
        avg_logits = logits_s.mean(0)                                    
        probs      = F.softmax(avg_logits, dim=1)
        p_sorted, _ = probs.sort(dim=1, descending=True)
        margins    = p_sorted[:, 0] - p_sorted[:, 1]                                 

        high_risk = margins < self.tau                                          

        if not high_risk.any():
            return avg_logits

                                                                    
        logits_l  = self.stochastic_forward(x, self.M_large)                 
        num_cls   = avg_logits.shape[1]
        votes     = torch.zeros(B, num_cls, device=x.device)                     
        preds_all = logits_l.argmax(dim=2)                                
        for preds in preds_all:                                              
            votes.scatter_add_(1, preds.unsqueeze(1),
                               torch.ones(B, 1, device=x.device))
                                                    
        avg_probs_l = F.softmax(logits_l.mean(0), dim=1)
        vote_scores = votes + avg_probs_l * 1e-3                                  

        result = avg_logits.clone()
        result[high_risk] = vote_scores[high_risk]
        return result
