# ISE train script for MSMT17

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u examples/ise_train_usl.py \
-b 256 \
-a resnet50 \
-d msmt17 \
--iters 400 \
--momentum 0.1 \
--eps 0.7 \
--num-instances 16 \
--use-hard \
--logs-dir ../logs/ISE_MSMT17/ \
--data-dir ../data/ \
--step-size 20 \
--epochs 50 \
--save-step 1 \
--eval-step 1 \
--sample-type ori \
--use_support \
--lp_loss_weight 0.01 \


