# ISE test script for Market1501
CUDA_VISIBLE_DEVICES=0 python -u examples/ise_test_usl.py \
-b 256 \
-a resnet50 \
-d market1501 \
--logs-dir ../logs/ISE_Market1501/test/ \
--data-dir ../data/ \
--resume ../model/ISE_M.pth.tar

