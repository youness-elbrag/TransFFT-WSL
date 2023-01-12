import torch.nn as nn 
import torch 
from blocks_networks import TransLayer
from Positional_encoding import FFTPEG , FF_ATPEG

class TransFFTMIL(pl.LightningModule):
    def __init__(self,n_classes,Positional_encoding):
        super(TransformMIL,self).__init__()
        self.Postional_Layer = Positional_encoding(12, 12, 512)
        self.fc_ = nn.Sequential(nn.Linear(1024,512),nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1,1,512))
        self.layer_ = TransLayer(dim=512)
        self.layer_1 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.Classifer= nn.Linear(512,n_classes)

    @TimeCounter_Process    
    def forward(self , **kwargs):
        h = kwargs['data'][0].float() #[Batch , N , 1024]
        h = self.fc_(h) ## [batch , n 512]
        print(h.shape)

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        
       #---->Translayer x1
        h = self.layer_(h) ## [batch , N , 512]
        #---->PPEG
        h = self.Postional_Layer(h, _H, _W) #[B, N, 512]
        #---->Translayer x2
        h = self.layer_1(h) #[B, N, 512]
        
        #---->cls_token
        h = self.norm(h)[:,0]
        
        #---->predict
        logits = self.Classifer(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits , Y_prob , Y_hat

if __name__ == '__main__':

    data= torch.randn((1,1, 2134,1024))
    model = TransFFTMIL(n_classes=2,Positional_encoding=FFTPEG)
    out = model(data=data)
    print(out)
    #model_.eval()
        