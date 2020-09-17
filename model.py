import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models import resnet34
import numpy as np
from PIL import Image
import scipy.signal as signal
import torch
import cv2
from ssim import SSIM

#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from torchvision import models


def ssimLoss(x,y):
    ssim_loss = SSIM()
    return -ssim_loss(x,y)

def MSE_Loss(x,y):
    mseloss=nn.MSELoss().cuda()
    return mseloss(x,y)

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda()
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda()
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std



def adain_mean(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    
    normalized_features = content_features+ style_mean
    return normalized_features

n_disc=16
img_size=256
class Dimg(nn.Module):
    def __init__(self):
        super(Dimg,self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(3,n_disc,4,2,1), #(3,16,4,2,1) 128
        )


        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc,n_disc*2,4,2,1), #(64,32,4,2,1)   (16,64,64,64,64)->(16,32,32,32) 64
            nn.ReLU(),

            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),#(16,32,32,32)->(16,64,16,16) 32 
            nn.ReLU(),

            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),#->(16,128,8,8) 16
            nn.ReLU(),
            nn.Conv2d(n_disc*8,n_disc*16,4,2,1),#->(16,128,8,8) 8
            nn.ReLU()
        )

        self.fc_common = nn.Sequential(
            nn.Linear(8*8*n_disc*16,1024),
            nn.ReLU()
        )
        self.fc_head1 = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )


    def forward(self,img):

        conv_img = self.conv_img(img)  # (20,3,128,128)->(20,16,64,64)


        total_conv = self.total_conv(conv_img).view(-1,8*8*n_disc*16)#(16,64,64,64)->32->16->8->(16,128,8,8)->(16,8192)
        body = self.fc_common(total_conv)#(16,8192)->(20,1024)

        head1 = self.fc_head1(body)#(16,1024)->(20,1)

        return head1
    



class VGGAttn(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
#        print(vgg)
        self.slice1 = vgg[: 2]  #64 256 256 
        self.slice2 = vgg[2: 7] #128 128 128
        self.slice3 = vgg[7: 12]   #256 64 64
        self.slice4 = vgg[12: 23]  #512 32 32
#        self.slice5 = vgg[21:30]
#        self.slice6 = vgg[30:]
#        self.conv = nn.Conv2d(512, 2048, 3,padding=1)

            
            
            
        vgg_2 = vgg19(pretrained=True).features
#        print(vgg)
        self.slice1_2 = vgg_2[: 2]  #64 256 256 
        self.slice2_2 = vgg_2[2: 7] #128 128 128
        self.slice3_2 = vgg_2[7: 12]   #256 64 64
        self.slice4_2 = vgg_2[12: 23]  #512 32 32
#        self.slice5 = vgg[21:30]
#        self.slice6 = vgg[30:]
#        self.conv = nn.Conv2d(512, 2048, 3,padding=1)
        for p in self.parameters():
            p.requires_grad = False
            
            
        self.coa_res2 = CoAttention(channel=256)
        self.coa_res3 = CoAttention(channel=512)
#        self.coa_res4 = CoAttention(channel=1024)
#        self.coa_res5 = CoAttention(channel=2048)


    def forward(self, content,style, output_last_feature=False):
        
        h1 = self.slice1(content)
        h1_2=self.slice1_2(style)
        
        h2 = self.slice2(h1)
        h2_2 = self.slice2_2(h1_2)
        h3 = self.slice3(h2)
        h3_2 = self.slice3_2(h2_2)
        
        h4_r=self.slice4(h3_2)
        
        
        Za, Zb, Qa, Qb = self.coa_res2(h3, h3_2)
        
        h3 = F.relu(Zb + h3+ Qb)
        h3_2 = F.relu(Qb + h3_2)
        
        
        h4 = self.slice4(h3)
        
        
        h4_2 = self.slice4_2(h3_2)

        
        
        Za, Zb, Qa, Qb = self.coa_res3(h4, h4_2)
        
        h4 = F.relu(Zb + h4)
        
        h4_2 = F.relu(Qb + h4_2)
        
        h4=adain_mean(h4,h4_r)

        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4
        
        
        
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        print(vgg)
        self.slice1 = vgg[: 2] #64
        self.slice2 = vgg[2: 7] #128
        self.slice3 = vgg[7: 12]# 256
        self.slice4 = vgg[12: 23]  #512 32 32
        for p in self.parameters():
            p.requires_grad = False


    def forward(self, images, output_last_feature=False):
        
        h1 = self.slice1(images)

        h2 = self.slice2(h1)

        h3 = self.slice3(h2)

        h4 = self.slice4(h3)

        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4       

        
        
        

      
class Decoder(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        self.rc1 = RC_res(512, 256, 3, 1)
        self.rc2 = RC_res(256, 256, 3, 1)
        self.rc3 = RC_res(256, 256, 3, 1)
        self.rc4 = RC_res(256, 256, 3, 1)
        self.rc5 = RC_res(256, 128, 3, 1)
        self.rc6 = RC_res(128, 128, 3, 1)
        self.rc7 = RC_res(128, 64, 3, 1)
        self.rc8 = RC_res(64, 64, 3, 1)
        self.rc9 = RC_res(64, 3, 3, 1, False)
    #
    #        self.sa1 = Self_Attn_Pro(64,'relu')
    #        self.sa2 = Self_Attn_Pro(64,'relu')
    
    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        #        h = self.sa1(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        #        h = self.sa2(h)
        h = self.rc9(h)
        #        print('h',h.size())
        return h
class CoAttention(nn.Module):
    def __init__(self, channel):
        super(CoAttention, self).__init__()

        d = channel // 16
        self.proja = nn.Conv2d(channel, d, kernel_size=1)
        self.projb = nn.Conv2d(channel, d, kernel_size=1)

        self.bottolneck1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.bottolneck2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.proj1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.proj2 = nn.Conv2d(channel, 1, kernel_size=1)

        self.bna = nn.BatchNorm2d(channel)
        self.bnb = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, Qa, Qb):

        # cascade 1
        Qa_1, Qb_1, Aa_1, Ab_1 = self.forward_sa(Qa, Qb) # Qa_attentin Qb_attention Aa Ab
        Za, Zb = self.forward_co(Qa_1, Qb_1)
        

        return Za, Zb, Qa_1, Qb_1


    def forward_sa(self, Qa, Qb):  
        Aa = self.proj1(Qa)
        Ab = self.proj2(Qb)

        n,c,h,w = Aa.shape
        Aa = Aa.view(-1, h*w) # (2 1024)
        Ab = Ab.view(-1, h*w)

        Aa = F.softmax(Aa, dim=-1)
        Ab = F.softmax(Ab, dim=-1)

        Aa = Aa.view(n,c,h,w)
        Ab = Ab.view(n,c,h,w)


        Qa_attened = Aa * Qa
        Qb_attened = Ab * Qb

        return Qa_attened, Qb_attened, Aa, Ab
    def forward_co(self, Qa, Qb): #Qa_attentin Qb_attention
        Qa_low = self.proja(Qa)
        Qb_low = self.projb(Qb)

        N, C, H, W = Qa_low.shape
        Qa_low = Qa_low.view(N, C, H * W)
        Qb_low = Qb_low.view(N, C, H * W)
        Qb_low = torch.transpose(Qb_low, 1, 2)

        L = torch.bmm(Qb_low, Qa_low)

        Aa = F.tanh(L)
        Ab = torch.transpose(Aa, 1, 2)

        N, C, H, W = Qa.shape

        Qa_ = Qa.view(N, C, H * W)
        Qb_ = Qb.view(N, C, H * W)

        Za = torch.bmm(Qb_, Aa)
        Zb = torch.bmm(Qa_, Ab)
        Za = Za.view(N, C, H, W)
        Zb = Zb.view(N, C, H, W)

        Za = F.normalize(Za)
        Zb = F.normalize(Zb)

        return Za, Zb


class RC_res(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batch = nn.BatchNorm2d(out_channels)
        self.activated = activated
        

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        h = self.batch(h)
        if self.activated:
            return F.relu(h)
        else:
            return h
