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


def apply_kcr_vgg16(model: 'VGG16', key: int) -> None:
    rng = torch.Generator()
    rng.manual_seed(key)

    def rperm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=rng)

    def rscale(n: int) -> torch.Tensor:
        return torch.rand(n, generator=rng) * 0.5 + 0.75

    layers = []
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            layers.append(m)
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            layers.append(m)

    cur_perm: torch.Tensor | None = None
    cur_scale: torch.Tensor | None = None

    with torch.no_grad():
        for idx, layer in enumerate(layers):
            is_last = (idx == len(layers) - 1)
            W     = layer.weight.data
            out_c = W.shape[0]
            in_c  = W.shape[1]

            if cur_perm is None:
                perm_in  = torch.arange(in_c)
                scale_in = torch.ones(in_c)
            elif in_c != cur_perm.shape[0]:
                C_prev = cur_perm.shape[0]
                HW     = in_c // C_prev
                perm_in  = torch.cat([cur_perm[c] * HW + torch.arange(HW)
                                      for c in range(C_prev)])
                scale_in = cur_scale.repeat_interleave(HW)
            else:
                perm_in  = cur_perm
                scale_in = cur_scale

            if is_last:
                perm_out  = torch.arange(out_c)
                scale_out = torch.ones(out_c)
            else:
                perm_out  = rperm(out_c)
                scale_out = rscale(out_c)

            layer.weight.data = _xform_weight(W, perm_out, scale_out, perm_in, scale_in)
            if layer.bias is not None:
                layer.bias.data = _xform_bias(layer.bias.data, perm_out, scale_out)

            cur_perm  = perm_out
            cur_scale = scale_out


class VGG16(nn.Module):
    def __init__(self, num_classes=100, defense=True, key=2025, bits=8):
        super(VGG16, self).__init__()
        self.defense = defense
        self.key = key
        self.bits = bits

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

                                                                                
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

        if self.defense:
            with torch.no_grad():
                apply_kcr_vgg16(self, self.key)
                for name, param in self.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        param.data = quantize_affine(param.data, bits=self.bits)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

        logits_s    = self.stochastic_forward(x, self.M_small)
        avg_logits  = logits_s.mean(0)
        probs       = F.softmax(avg_logits, dim=1)
        p_sorted, _ = probs.sort(dim=1, descending=True)
        margins     = p_sorted[:, 0] - p_sorted[:, 1]

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
