import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
import math

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.feature = nn.Sequential(*list(models.efficientnet_b0(weights='DEFAULT').children())[:-2],
                                     nn.Conv2d(1280, 128, 3,1,1),
                                    )
        self.classifier = nn.Sequential(
            nn.Linear(128, 100),
            nn.PReLU(),
            nn.Linear(100, 2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        conv_feature_map = self.feature(x) # torch.Size([1, 1792, 7, 7])
        conv_feature_map = self.avg_pool(conv_feature_map).view(conv_feature_map.size(0), -1)
        logits = self.classifier(conv_feature_map)
        return conv_feature_map, logits

# ==============================================================
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

    
def conv_bn(input, output, kernel_size, stride): 
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride, kernel_size//2, bias=False),
        nn.BatchNorm2d(output),
        SiLU()
    )
    
class MBConv(nn.Module):
    def __init__(self, input, output, stride, expand_ratio, use_se) -> None:
        super().__init__()
        assert stride in [1, 2]

        hidden_dimension = round(input * expand_ratio)
        self.identity = (stride == 1) and (input == output)

        if use_se: #MBCONV
            self.conv = nn.Sequential(
                nn.Conv2d(input, hidden_dimension, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),
                ECAAttention(kernel_size=3),
                nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output)
            )
        else: #Fused-MBConv
            self.conv = nn.Sequential(
                nn.Conv2d(input, hidden_dimension, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),
                nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),
                ECAAttention(kernel_size=3),
                nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output)
            )
            

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ECAAttention(nn.Module):
    '''
    Efficient Channel Attention
    '''
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

        

class lightNN(nn.Module):
    def __init__(self, cfgs=None, in_channel=3, num_classes=2, width_multiplier=1):
        super().__init__()
        # t: expansion ratio
        # c: output channel
        # n: block repeat times
        # s: stride
        # UMB: use fused-MBConv or not

        cfgs = [
            # t, c, n, s, use_fusedMBConv (0: don't use)
            [2, 8, 6, 1, 0],
            [2, 16, 4 , 1, 0]
        ]
        # cfgs = [
        #     # t, c, n, s, use_fusedMBConv (0: don't use)
        #     [1,  8,  1, 1, 1],
        #     [1,  16,  1, 1, 1],
        #     [1,  32,  1, 1, 1],
        #     [1,  64,  1, 1, 1],
        # ]
        self.cfgs = cfgs

        input_channel = make_divisible(8 * width_multiplier, 8)
        layers = [conv_bn(in_channel, input_channel,3,2)]

        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = make_divisible(c * width_multiplier, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, (s if i==0 else 1), t, use_se))
                # (input, output, stride, expand_ratio, use_se)
                input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        output_channel = make_divisible(128 * width_multiplier, 8) if width_multiplier > 1.0 else 128
        self.conv = conv_bn(input_channel, output_channel, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        feature_maps = x.view(x.size(0), -1)
        logits = self.classifier(feature_maps)
        return  feature_maps, logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    
if __name__ == '__main__':
    # calulate flops and params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep = DeepNN().cuda()
    light = lightNN().cuda()
    input = torch.randn(1, 3,185, 160).cuda()
    flops, params = profile(deep, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))
    input = torch.randn(1, 3, 185, 160).cuda()
    flops, params = profile(light, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))