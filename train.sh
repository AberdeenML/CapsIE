#!/bin/bash
python main.py --experience SIECaps --exp-dir './exps/SIECaps/' --root-log-dir './logs/' --epochs 2000 --arch resnet18 \
    --equi 256 \
    --batch-size 1024 \
    --base-lr 1e-3 \
    --dataset-root /path/to/dataset/3DIEBench/ \
    --images-file ./data/train_images.npy \
    --labels-file ./data/train_labels.npy \
    --sim-coeff 10 \
    --std-coeff 10 \
    --cov-coeff 1 \
    --temp_init 0.1 0.025 \
    --temp_end 0.1 0.1 \
    --scale-equi \
    --mlp 1111-16-16 \
    --caps-type SR \
    --caps-depth 1 \
    --equi-factor 0.45 \
    --hypernetwork linear \
    --resolution 256
    
    
