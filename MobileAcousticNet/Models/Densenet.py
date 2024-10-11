from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

# 由于在提取梅尔特征时，只用了13各三角滤波器，所以特征图只有13行，第一维的维度较小
# 所以很网络的层数不宜过深，这样会导致feature map被压缩到没有
# 所以在做对比实验的时候，不要选择层数过深的网络模型来进行对比
# 进一步说，在音频检测这个任务中，是不是提高网络的宽度比单纯的一味提高网络的深度效果要来的好？！！！！！！！！！！！

#### 针对要进行的对比实验，由于网络层数过深会导致第一个维度降为0，
#### 所以对Desennet的层数做了自定义的调整

## 这个Densenet模型参数量居然非常少，所以后续应该还要看计算量
## 而且Densenet在rain这环境中的效果居然出奇的好，所以在与densennet进行对比讨论的时候
### 应该着重于分析模型的泛用性和运算量

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)
        # 在通道维上将输入和输出连结
        return torch.cat([x, new_features], 1)
class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)
class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=1))
class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(2, 4, 6, 4), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        """
        :param growth_rate: 增长率，即K=32
        :param block_config: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param num_init_features: 第一个卷积的通道数一般为2*K=64
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param compression_rate: 压缩因子
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(2, num_init_features, kernel_size=5, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        features = self.features(x)
        # print(features.shape)
        out = F.adaptive_avg_pool2d(features,output_size=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
myModel = DenseNet(num_classes=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# print(myModel)
next(myModel.parameters()).device

# from thop import profile
# from thop import clever_format
# input=torch.randn(1,2,13,403).cuda()
# flops, params = profile(myModel, inputs=(input,))
# print(flops, params)
# flops, params = clever_format([flops, params], "%.3f")
# print(flops, params)