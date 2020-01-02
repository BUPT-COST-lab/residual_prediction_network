#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='results'
modelpath='checkpoints'
bak_path='checkpoints/bak'
logname='logs/train_residual_net.log'
nohup python -u train_ResNet_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m20_train_3d.npy \
    --test_data_paths data/test20_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
    --input_length 10 \
    --seq_length 20 \
    --stacklength 6 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 4 \
    --snapshot_interval 500  >${logname}  2>&1 &
tail -f ${logname}




