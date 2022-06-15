CUDA_VISIBLE_DEVICES=0 python -u train.py \
 --database SJTU \
 --model_name  ResNet_mean_with_fast \
 --split_num 9 \
 --conv_base_lr 0.00004 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 6 \
 --epochs 30 \
 --split_num 9 \
 --crop_size 224 \
 --frame_index 5 \
 --ckpt_path ckpts \
 >> logs/sjtu.log  