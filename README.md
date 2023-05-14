本仓库主要用于CV方向的深度学习记录，实现主要的神经网络，及各种学习任务

## 训练模型
```python
python train.py
```

## 模型推理
```python
python predict.py
```

## 支持功能
- [x] 图像分类
![inference](classification/assets/infer_animal100.png)
- [ ] 目标检测
- [ ] 语义分割
- [ ] 实例分割
- [ ] ......

## 模型进度
- 图像分类
- [x] [LeNet](classification/backbone/alexnet.py)
- [x] [AlexNet](classification/backbone/alexnet.py)
- [x] [VGG](classification/backbone/vgg.py)  
- [x] [GoogleNet](classification/backbone/googlenet.py)
- [x] [ResNet](classification/backbone/resnet.py)
- [x] [MobileNet](classification/backbone/shufflenet.py)
- [ ] DenseNet

  ...
 
## 模型下载
- [LeNet-epoch100-cifar10](https://deepl-ckpt-classification.gd2.qingstor.com/lenet/lenet_cifar10_epoch_100.pth)
- ......