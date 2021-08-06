export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune_imagenet.py     \
 -a qmobilenetv2                 \
 -c checkpoints/finetune_mobilenetv2    \
 --data_name imagenet          \
 --data data/imagenet/           \
 --epochs 32                     \
 --lr 0.01                    \
 --pretrained       \
 --gpu_id 0,1,2,3     \
 --train_batch_per_gpu 96              \
 --wd 4e-5                       \
 --workers 32                    \