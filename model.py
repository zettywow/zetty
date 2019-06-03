from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from torch.autograd import Variable
import torch.nn as nn
################################## Arcface resnet_ir_se ########################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockIR(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(BasicBlockIR, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU()               #self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LResNetIR(nn.Module):
    def __init__(self, block, layers, use_se=False):  #layers=[2,2,2,2], [3,4,6,3],.....
        super(LResNetIR, self).__init__()

        self.inplanes = 64
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)    #Input:Bx3x112x96-->Output: Bx64x112x96
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU()  #nn.ReLU(inplace=True)

        #maxpool is deleted and downsampling is now replaced by self.layer1 with stride-2
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #3x3 max-pool,/2: Bx64x56x56
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)  #Bx64x56x48
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #Bx128x28x24
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #Bx256x14x12
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #Bx512x7x6

        self.bn_d = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.5)
        self.fc_f = nn.Linear(512*7*7, 512)
        self.bn_f = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_d(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc_f(x)
        x = self.bn_f(x)

        # L2-normalize input: BxD = Bx512
        #x = F.normalize(x, p=2, dim=1)
        return x

def LResNet50IR(use_se = False, **kwargs):  #LResNet50A()
    """Constructs a LResNet-50A model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LResNetIR(BasicBlockIR, [3, 4, 14, 3], use_se = use_se, **kwargs)
    #model = LResNetIR(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def LResNet18IR(use_se=False, **kwargs):
    model = LResNetIR(BasicBlockIR, [2, 2, 2, 2], use_se = use_se, **kwargs)
    return model

##################################  Arcface fc #############################################################

class Arcface(nn.Module):
    def __init__(self, embedding_size=512, classnum=0, s=64.0,m = 0.5,easy_margin=False):
        super(Arcface, self).__init__()
        self.in_dim = embedding_size
        self.out_dim = classnum
        self.weight = Parameter(torch.FloatTensor(self.out_dim,self.in_dim))
        #torch.Tensor.renorm(p, dim, maxnorm)
        #self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.xavier_uniform_(self.weight)
        #self.relu = nn.ReLU()

        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        # print(output)

        return output

class Arcface_insight(nn.Module):
    def __init__(self, embedding_size=512, classnum=0, s=64.0,m = 0.5,easy_margin=False):
        super(Arcface_insight, self).__init__()
        self.in_dim = embedding_size
        self.out_dim = classnum
        self.weight = Parameter(torch.FloatTensor(self.in_dim,self.out_dim))
        nn.init.xavier_uniform_(self.weight)

        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        w_norm = F.normalize(self.weight, p=2, dim=0)
        input_norm = F.normalize(input, p=2, dim=1)
        fc_out = input_norm.mm(w_norm) * self.s

        one_hot = torch.zeros(fc_out.size(),device = 'cuda')
        one_hot.scatter_(1,label.view(-1,1).long(),1)
        
        zy = torch.masked_select(fc_out,one_hot.byte()).view(-1,1)
        assert zy.shape == label.view(-1,1).shape
        assert zy[0] == fc_out[0,label.view(-1)[0]]
        
        cosine = zy/self.s
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        new_zy = cosine * self.cos_m - sine * self.sin_m
        new_zy = new_zy * self.s

        if self.easy_margin:
            new_zy = torch.where(cosine > 0, new_zy, zy)
        else:
            new_zy = torch.where(cosine > self.th, new_zy, zy - self.mm * self.s)

        diff = (new_zy - zy)
        body = one_hot.float() * diff
        fc_out = fc_out + body
        
        return fc_out
        
##################################  Margin loss  #############################################################  

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

##################################  Cosface fc #############################################################    
    
class Am_softmax(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369
    def forward(self,embbedings,label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

##################################  Triplet loss  #############################################################

class Tripletloss(nn.Module):

    def __init__(self,triplet_alpha = 0.3,beta = 0.2):
        super(Tripletloss, self).__init__()
        self.triplet_alpha = triplet_alpha
        self.beta = beta
        self.relu = nn.ReLU()

    def forward(self,input):
        input_norm = F.normalize(input, p=2, dim=1)
        batch = input_norm.shape[0]//3
        anchor = input_norm[0:batch]
        positive = input_norm[batch:2*batch]
        negative = input_norm[2*batch:]
        assert negative.shape[0] == positive.shape[0]

        ap = torch.sum(anchor * positive,dim=1,keepdim=True)
        an = torch.sum(anchor * negative,dim=1,keepdim=True)
        #ap = ap*ap
        #an = an*an
        #ap = torch.sum(ap,dim=1,keepdim=True)
        #an = torch.sum(an,dim=1,keepdim=True)
        loss_c = (1-ap).mean()
        loss_t = self.relu(an-ap+self.triplet_alpha).mean()
        loss = loss_t + self.beta * loss_c
        return loss


