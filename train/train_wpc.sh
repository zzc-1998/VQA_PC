CUDA_VISIBLE_DEVICES=0 python -u train.py \
 --database WPC \
 --model_name  ResNet_mean_with_fast \
 --conv_base_lr 0.00005 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --epochs 30 \
 --split_num 5 \
 >> logs/wpc.log  

