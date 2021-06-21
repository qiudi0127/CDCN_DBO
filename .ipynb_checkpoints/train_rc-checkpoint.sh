#!/bin/bash sh 
python -u train_BCNpp_model.py \
    --gpu '0,1,2,3,4,5,6,7'  \
    --lr 0.0001 \
    --batchsize 20 \
    --step_size 20 \
    --gamma 0.5 \
    --echo_batches 20 \
    --epochs 400 \
    --log BCN_128 \
    --theta 0.5 \
    --map_size 16 \
    --crop_size 128 \
    --train_list 'dataset/train_label_balance.txt'  \
    --val_list 'dataset/val_label.txt'  \
    --root 'dataset' \
    --colorpath '/home/hadoop-fincv/cephfs/data/zhengmingwu/iccv_workshops/color.h5' \
    --illupath '/home/hadoop-fincv/cephfs/data/zhengmingwu/iccv_workshops/illu.h5' \
    --train_ldmk 'train_landmark' \
    --val_ldmk 'val_landmark' \
    --binary_type 'gray'

#     --train_list '/ssd/zhengmingwu/iccv_dataset/train_label_balance.txt'  \
#     --val_list '/ssd/zhengmingwu/iccv_dataset/val_label.txt'  \
#     --root '/ssd/zhengmingwu/iccv_dataset/' \
#     --colorpath '/ssd/zhengmingwu/iccv/color.h5' \
#     --illupath '/ssd/zhengmingwu/iccv/illu.h5' \