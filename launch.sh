#! /bin/bash

export WORKSPACE="/checkpoint/wesbz/datasets01/audioset_hdf5"

#python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=400 --hop_size=160 --mel_bins=80 --fmin=0 --fmax=8000 --model_type='TransformerModel' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=32 --learning_rate=1e-4 --resume_iteration=0 --early_stop=1000000 --cuda
python3 pytorch/main.py train --workspace=$WORKSPACE --data_type='full_train' --window_size=1024 --hop_size=320 --mel_bins=80 --sample_rate=48000 --fmin=0 --fmax=24000 --model_type='TransformerModel' --loss_type='clip_bce' --balanced='balanced' --augmentation='mixup' --batch_size=1280 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda
