This repository provides the implementation of the following paper:

> Y. Bo, M. Tao, and K. Niu, “Distributed Semantic Communication via Nonlinear Transform Coding for Correlated Images,” *IEEE Transactions on Communications*, vol. 74, pp. 9669–9684, 2026.

The code supports two methods:

- **D-NTSCC**: Distributed nonlinear transform source-channel coding
- **D-NTSC**: Distributed nonlinear transform source coding

Experiments can be conducted on the **KITTI** and **Cityscapes** correlated image datasets over AWGN or fading channels.

## Training

### 1. Prepare the Dataset

Place the datasets in the project directory:

```text
project/
├── KITTI/
├── Cityscapes/
├── main.py
├── train.py
├── network.py
├── network_sc.py
├── kitti.py
├── cityscapes.py
└── utils.py
```

The dataset directories should contain the corresponding training and test data required by `PairKitti` or `PairCityscape`.

### 2. Train D-NTSCC

Run the following command to train D-NTSCC on KITTI:

```bash
python main.py \
  --method d-ntscc \
  --dataset kitti \
  --channel_mode awgn \
  --loss psnr
```

To train on Cityscapes:

```bash
python main.py \
  --method d-ntscc \
  --dataset cityscapes \
  --channel_mode awgn \
  --loss psnr \
```

### 3. Train D-NTSC

Run the following command to train D-NTSC:

```bash
python main.py \
  --method d-ntsc \
  --dataset kitti \
  --loss psnr \
```

## Requirements

Install the required Python packages before training:

```bash
pip install -r requirements.txt
```
