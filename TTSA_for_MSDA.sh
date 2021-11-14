#!/usr/bin/env bash

# Office-Home
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet50 --seed 2 --dset office-home --root /data1/TL/data/office-home-65/ --output_dir log/TTSA_for_MSDA/home --target a --epochs 40 --iters-per-epoch 1000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet50 --seed 2 --dset office-home --root /data1/TL/data/office-home-65/ --output_dir log/TTSA_for_MSDA/home --target c --epochs 40 --iters-per-epoch 1000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet50 --seed 2 --dset office-home --root /data1/TL/data/office-home-65/ --output_dir log/TTSA_for_MSDA/home --target p --epochs 40 --iters-per-epoch 1000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet50 --seed 2 --dset office-home --root /data1/TL/data/office-home-65/ --output_dir log/TTSA_for_MSDA/home --target r --epochs 40 --iters-per-epoch 1000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1


# DomainNet
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target c --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target i --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target p --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target q --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target r --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_MSDA.py --gpu_id 0 --arch resnet101 --seed 2 --center_crop --dset DomainNet --root /data1/TL/data/list/domainnet --output_dir log/TTSA_for_MSDA/domainnet  --target s --epochs 30 --iters-per-epoch 2000 --lambda0 0.25 --lr 0.01  --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1

