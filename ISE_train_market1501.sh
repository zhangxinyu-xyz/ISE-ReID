# ISE train script for Market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u examples/ise_train_usl.py \
-b 256 \
-a resnet50 \
-d market1501 \
--iters 400 \
--momentum 0.2 \
--eps 0.4 \
--num-instances 16 \
--use-hard \
--logs-dir ../logs/ISE_Market1501/ \
--data-dir ../data/ \
--step-size 30 \
--epochs 70 \
--save-step 1 \
--eval-step 1 \
--sample-type hard \
--use_support \
--lp_loss_weight 0.1 \




