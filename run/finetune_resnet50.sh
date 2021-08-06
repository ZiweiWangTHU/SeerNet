export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune_imagenet.py     \
 -a qresnet50                 \
 -c checkpoints/train_resnet50    \
 --data_name imagenet          \
 --data data/imagenet/           \
 --epochs 100                     \
 --lr 0.01                    \
 --gpu_id 0,1,2,3     \
 --pretrained   \
 --train_batch_per_gpu 96              \
 --wd 4e-5                       \
 --workers 32                    \