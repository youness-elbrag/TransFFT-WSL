import torch.nn as nn 
import torch 
from blocks_networks import SpectralConv2d , Attention_block
from utilits import TimeCounter_Process

class FF_ATPEG(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FF_ATPEG, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.attetion_gate = Attention_block()
        self.padding = 9 # pad the domain if input is non-periodic
        #self.fc0 = nn.Linear(512,512 ) # input channel is 3: (a(x, y), x, y)
        self.conv0  = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.projection_= nn.Conv2d(self.width, self.width, 7, 1, 7//2, groups=self.width)
        self.projection_1 = nn.Conv2d(self.width, self.width, 5, 1, 5//2, groups=self.width)
        self.projection_2 = nn.Conv2d(self.width, self.width, 3, 1, 3//2, groups=self.width)
    @TimeCounter_Process    
    def forward(self,x ,H, W):
            B,_,C = x.shape
            cls_token,feat_token = x[:,0],x[:, 1:]
            cnn_feat = feat_token.transpose(1,2).view(B,C,H,W)

            x1 =  self.conv0(cnn_feat) 
            x2 =  self.projection_(x1)
            x =  x1+x2
            x = F.gelu(x)

            x3 =  self.conv1(cnn_feat) 
            x4 =  self.projection_1(x3)
            x =self.attetion_gate(x4,x1)## x1+x2
            x = F.gelu(x)

            x5 =  self.conv2(x) 
            x6 =  self.projection_2(x5)
            x = self.attetion_gate(x6,x3) # ## x1+x2
            x = F.gelu(x)

            x = x.flatten(2).transpose(1,2)
            x = torch.cat((cls_token.unsqueeze(1),x),dim=1)       
            return x

class FFTPEG(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FFTPEG, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        #self.fc0 = nn.Linear(512,512 ) # input channel is 3: (a(x, y), x, y)
        self.conv0  = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.projection_= nn.Conv2d(self.width, self.width, 7, 1, 7//2, groups=self.width)
        self.projection_1 = nn.Conv2d(self.width, self.width, 5, 1, 5//2, groups=self.width)
        self.projection_2 = nn.Conv2d(self.width, self.width, 3, 1, 3//2, groups=self.width)
    @TimeCounter_Process    
    def forward(self,x ,H, W):
        B,_,C = x.shape
        cls_token,feat_token = x[:,0],x[:, 1:]
        cnn_feat = feat_token.transpose(1,2).view(B,C,H,W)
        x =  self.conv0(self.projection_(cnn_feat))+cnn_feat+self.conv1(self.projection_1(cnn_feat))+self.conv2(self.projection_2(cnn_feat))
        x = x.flatten(2).transpose(1,2)
        x = torch.cat((cls_token.unsqueeze(1),x),dim=1)       
        return x






