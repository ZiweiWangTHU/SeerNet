
# SeerNet

​	This is the pytorch implementation for the paper: *Learning Accurate Performance Predictors for Ultrafast Automated Model Compression*, which is in submission to IJCV. This repo contains active sampling for training the performance predictor, optimizing the compression policy and finetuning on two datasets(VGG-small, ResNet20 on Cifar-10; ResNet18, MobileNetv2, ResNet50 on ImageNet) using our proposed SeerNet.

​	As for the entire pipeline, we firstly get a few random samples to pretrain the MLP predictor. After getting the pretrained predictor, we execute active sampling using evolution search to get samples, which are used to further optimize the predictor above. Then we search for optimal compression policy under given constraint utilizing the predictor. Finally, we finetune the policy until convergence.

## Quick Start

### Prerequisites

- python>=3.5
- pytorch>=1.1.0
- torchvision>=0.3.0 
- other packages like numpy and sklearn

### Dataset

If you already have the ImageNet dataset for pytorch, you could create a link to data folder and use it:
```
# prepare dataset, change the path to your own
ln -s /path/to/imagenet/ data/
```
If you don't have the ImageNet, you can use the following script to download it: 
[https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Active Sampling

You can run the following command to actively search the samples by evolution algorithm:

```
CUDA_VISIBLE_DEVICES=0 python PGD/search.py --sample_path=results/res18/resnet18_sample.npy --acc_path=results/res18/resnet18_acc.npy --lr=0.2 --batch=400 --epoch=1000 --save_path=search_result.npy --dim=57
```

## Training performance predictor

You can run the following command to training the MLP predictor:

```
CUDA_VISIBLE_DEVICES=0 python PGD/regression/regression.py --sample_path=../results/res18/resnet18_sample.npy --acc_path=../results/res18/resnet18_acc.npy --lr=0.2 --batch=400 --epoch=5000 --dim=57
```



## Compression Policy Optimization

After training the performance predictor, you can run the following command to optimize the compression policy:
```

# for resnet18, please use
python PGD/pgd_search.py --arch qresnet18 --layer_nums 19 --step_size 0.005 --max_bops 30 --pretrained_weight path\to\weight 


# for mobilenetv2, please use
python PGD/pgd_search.py --arch qmobilenetv2 --layer_nums 53 --step_size 0.005 --max_bops 8 --pretrained_weight path\to\weight 


# for resnet50, please use
python PGD/pgd_search.py --arch qresnet50 --layer_nums 52 --step_size 0.005 --max_bops 65 --pretrained_weight path\to\weight 

```

### Finetune Policy

After optimizing, you can get the optimal quantization and pruning strategy list, and you can replace the strategy list in finetune_imagenet.py to finetune and evaluate the performance on ImageNet dataset. You can also use the default strategy to reproduce the results in our paper.

For finetuning ResNet18 on ImageNet, please run:

```
bash run/finetune_resnet18.sh
```

For finetuning MobileNetv2 on ImageNet, please run:

```
bash run/finetune_mobilenetv2.sh
```
For finetuning ResNet50 on ImageNet, please run:

```
bash run/finetune_resnet50.sh
```

