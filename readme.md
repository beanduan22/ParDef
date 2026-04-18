# pardef

## Requirements

```bash
pip install -r requirements.txt
```

For Tiny-ImageNet, download the dataset and place it at `tinyimagenet/resnet32/data/tiny-imagenet-200` and `tinyimagenet/vgg16/data/tiny-imagenet-200`.

## Training

Train and apply the defense for each dataset and architecture:

```bash
cd cifar10
python train.py --arch resnet32
python train.py --arch vgg16

cd ../cifar100
python train.py --arch resnet32
python train.py --arch vgg16

cd ../tinyimagenet
python train.py --arch resnet32
python train.py --arch vgg16
```

Checkpoints are saved to `{arch}/checkpoint/`. The defended model is saved as `defended.pth`.

## Validation

Evaluate the defended model and compute the ARI threshold:

```bash
cd cifar10
python validate.py --arch resnet32
python validate.py --arch vgg16

cd ../cifar100
python validate.py --arch resnet32
python validate.py --arch vgg16

cd ../tinyimagenet
python validate.py --arch resnet32
python validate.py --arch vgg16
```

## Attacks

Each attack directory is named `{attack}/{arch}-{dataset}/` and contains five scripts. Run from inside the directory.

### ProFlip

```bash
cd ProFlip/resnet32-cifar10
python nonadaptive.py
python adaptive.py
python trigger_nonadaptive.py
python trigger_adaptive.py
```

### P3A

```bash
cd P3A/resnet32-cifar10
python nonadaptive.py
python adaptive.py
python trigger_nonadaptive.py
python trigger_adaptive.py
```

### APA

```bash
cd APA/resnet32-cifar10
python nonadaptive.py
python adaptive.py
python trigger_nonadaptive.py
python trigger_adaptive.py
```

Replace `resnet32-cifar10` with any of:
- `resnet32-cifar100`
- `resnet32-tinyimagenet`
- `vgg16-cifar10`
- `vgg16-cifar100`
- `vgg16-tinyimagenet`

Results are saved to `./result/` inside each attack directory.
