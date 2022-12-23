# ISE test script for MSMT17
CUDA_VISIBLE_DEVICES=0 python -u examples/ise_test_usl.py \
-b 256 \
-a resnet50 \
-d msmt17 \
--logs-dir ../logs/ISE_MSMT17/test/ \
--data-dir ../data/ \
--resume ../model/ISE_MS.pth.tar

