#!/usr/bin/env bash

# PACS
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet18 --target_domain art_painting --epochs 50 --schdule_gamma 1 --lr 0.02 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet18 --target_domain cartoon --epochs 50 --schdule_gamma 1 --lr 0.02 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet18 --target_domain photo --epochs 50 --schdule_gamma 1 --lr 0.02 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet18 --target_domain sketch --epochs 50 --schdule_gamma 1 --lr 0.02 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1

python3 train_TTSA_for_DG.py --gpu 1 --arch resnet50 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet50 --target_domain art_painting --epochs 50 --schdule_gamma 1 --lr 0.015 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet50 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet50 --target_domain cartoon --epochs 50 --schdule_gamma 1 --lr 0.015 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet50 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet50 --target_domain photo --epochs 50 --schdule_gamma 1 --lr 0.015 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet50 --batch_size 32 --data_dir /data1/TL/data/PACS/kfold/ --dset PACS --output_dir log/TTSA_for_DG/PACS/resnet50 --target_domain sketch --epochs 50 --schdule_gamma 1 --lr 0.015 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1

# Office-Home
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/office-home/ --dset office-home --output_dir log/TTSA_for_DG/home/resnet18 --target_domain Art --epochs 30 --gamma 10 --lr 0.005 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/office-home/ --dset office-home --output_dir log/TTSA_for_DG/home/resnet18 --target_domain Clipart --epochs 30 --gamma 10 --lr 0.005 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/office-home/ --dset office-home --output_dir log/TTSA_for_DG/home/resnet18 --target_domain Product --epochs 30 --gamma 10 --lr 0.005 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1
python3 train_TTSA_for_DG.py --gpu 1 --arch resnet18 --batch_size 32 --data_dir /data1/TL/data/office-home/ --dset office-home --output_dir log/TTSA_for_DG/home/resnet18 --target_domain RealWorld --epochs 30 --gamma 10 --lr 0.005 --lambda0 0.25 --gamma 0.001 --eta 0.5 --alpha 0.1 --beta 0.1











