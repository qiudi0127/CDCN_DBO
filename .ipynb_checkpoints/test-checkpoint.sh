#!/bin/bash sh 
python -u test_BCNpp_model_fin.py \
    --gpu '0,1,2,3' \
    --epoch 157 \
    --resume 'BCN_128_gray_aug/BCNpp_128_gray_aug_157.pth' \
    --log 'test_result/BCN_test' \
    --image_dir 'dataset/test/' \
    --ldmk_dir 'dataset/test_landmark_R50'
    
