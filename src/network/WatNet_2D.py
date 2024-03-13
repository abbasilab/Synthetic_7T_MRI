import numpy as np
import torch
import torch.nn as nn
from pywt import dwt2


def wat_3(batch_input):
    a,b,c,d=batch_input.cpu().numpy().shape

    wat1=[]
    wat2=[]
    wat3=[]
    for a in range(batch_input.cpu().numpy().shape[0]):
        iwat1=[]
        iwat2=[]
        iwat3=[]
        for i in range(b):
            img_p=batch_input.cpu().numpy()[a,i,:,:]
            cA1, (cH1, cV1, cD1) = dwt2(img_p, 'haar') 
            cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar') 
            cA3, (cH3, cV3, cD3) = dwt2(cA2, 'haar')

            iwat1.append([cH1,cV1,cD1])
            iwat2.append([cH2,cV2,cD2])
            iwat3.append([cA3,cH3,cV3,cD3])

        iwat1=np.array(iwat1).reshape([1,b*3,int(c/2),int(d/2)])
        iwat2=np.array(iwat2).reshape([1,b*3,int(c/4),int(d/4)])
        iwat3=np.array(iwat3).reshape([1,b*4,int(c/8),int(d/8)])

        wat1.append(iwat1)
        wat2.append(iwat2)
        wat3.append(iwat3)

    wat1=torch.tensor(np.squeeze(wat1)).to('cuda')
    wat2=torch.tensor(np.squeeze(wat2)).to('cuda')
    wat3=torch.tensor(np.squeeze(wat3)).to('cuda')
    return wat1,wat2,wat3

class conv_init3(nn.Module):
    #3 slice input
    def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.net(x)
    
class DBlock(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(48, 32, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv2d(48, 64, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(48, 80, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv2d(80, 64, (1, 1), stride=2, padding=0)

    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        return x1
    
class DBlock_interim(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(48, 32, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv2d(48, 64, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(48, 80, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv2d(80, 64, (1, 1), stride=1, padding=0)
            self.relu=nn.ReLU()
            
    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        x1=self.relu(x1)
        return x1
    
class DBlock_recon(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(48, 32, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv2d(48, 64, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(64, 48, (3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(48, 80, (3, 3), stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv2d(80, 64, (1, 1), stride=1, padding=0)
            self.relu=nn.ReLU()
            self.upsample=nn.ConvTranspose2d(64,64,(4,4), stride=2, padding=1)
            
    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        x1=self.relu(x1)
        x1=self.upsample(x1)
        return x1
    
class wat_layer_3_12(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(9, 32, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x
    
class wat_layer_3_3(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(12, 32, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x
    
class wat_layer_1_12(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x
    
class wat_layer_1_3(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, (3, 3), stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x


class WatNet2D(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        self.c3 = conv_init3()
        self.d1 = DBlock()
        self.wat12 = wat_layer_3_12()
        self.wat3 = wat_layer_3_3()
        self.inter = DBlock_interim()
        self.recon = DBlock_recon()
        self.final_up = nn.Conv2d(64, 1, (17, 17), stride=1, padding=8)

    def forward(self, x):
        watp1, watp2, watp3 = wat_3(x)
        t = self.c3(x)
        t = self.d1(t)
        t = self.wat12(t, watp1)
        t = self.d1(t)
        t = self.wat12(t, watp2)
        t = self.d1(t)
        t = self.wat3(t, watp3)
        t = self.inter(t)
        t = self.inter(t)
        t = self.recon(t)
        t = self.recon(t)
        t = self.recon(t)
        t = self.final_up(t)
        t = torch.add(t, x)
        return t
