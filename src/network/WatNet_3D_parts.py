import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from pywt import dwtn, dwt2

def wat_3_3D(batch_input, waven):
    # channel size has to equal 1, only 1 channel allowed
    wat1=[]
    wat2=[]
    wat3=[]
    
    for a in range(batch_input.cpu().numpy().shape[0]):
        img_p=batch_input.cpu().numpy()[a,0,:,:,:]
        s=img_p.shape
        
        w1=dwtn(img_p,  waven) 
        for k in list(w1.keys()):
            s1=w1[k].shape
            w1[k] = zoom(w1[k], (s[0]/2/s1[0], s[1]/2/s1[1], s[2]/2/s1[2]))
        w2=dwtn(w1['aaa'],  waven) 
        for k in list(w2.keys()):
            s2=w2[k].shape
            w2[k] = zoom(w2[k], (s[0]/4/s2[0], s[1]/4/s2[1], s[2]/4/s2[2]))
        w3=dwtn(w2['aaa'],  waven) 
        for k in list(w3.keys()):
            s3=w3[k].shape
            w3[k] = zoom(w3[k], (s[0]/8/s3[0], s[1]/8/s3[1], s[2]/8/s3[2]))
            
        w1=np.array(list(w1.items()),dtype=object)
        w1=np.array([[w1[i][1] for i in range(1,8)]])
        w2=np.array(list(w2.items()),dtype=object)
        w2=np.array([[w2[i][1] for i in range(1,8)]])
        w3=np.array(list(w3.items()),dtype=object)
        w3=np.array([[w3[i][1] for i in range(0,8)]])

        wat1.append(w1)
        wat2.append(w2)
        wat3.append(w3)

    wat1=torch.tensor(np.squeeze(wat1)).to('cuda')
    wat2=torch.tensor(np.squeeze(wat2)).to('cuda')
    wat3=torch.tensor(np.squeeze(wat3)).to('cuda')
    return wat1,wat2,wat3

class conv_init_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 64, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.net(x)
    
class DBlock_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv3d(64, 48, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(48, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv3d(48, 64, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(64, 48, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(48, 80, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv3d(80, 64, 1, stride=2, padding=0)

    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        return x1
    
class DBlock_interim_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv3d(64, 48, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(48, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv3d(48, 64, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(64, 48, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(48, 80, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv3d(80, 64, 1, stride=1, padding=0)
            self.relu=nn.ReLU()
            
    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        x1=self.relu(x1)
        return x1
    
class DBlock_recon_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.before_split = nn.Sequential(
                nn.Conv3d(64, 48, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(48, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.after_split = nn.Sequential(
                nn.Conv3d(48, 64, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(64, 48, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv3d(48, 80, 3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.down=nn.Conv3d(80, 64, 1, stride=1, padding=0)
            self.relu=nn.ReLU()
            self.upsample=nn.ConvTranspose3d(64,64,4, stride=2, padding=1)
            
    def forward(self, x):
        x1=self.before_split(x)
        x1_16,x1=x1.split([16,48],dim=1)
        x1=self.after_split(x1)
        x1=torch.add(x1,torch.cat((x,x1_16),dim=1))
        x1=self.down(x1)
        x1=self.relu(x1)
        x1=self.upsample(x1)
        return x1
    
class wat_layer_1_12_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(7, 32, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x
    
class wat_layer_1_3_3D(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(8, 32, 3, stride=1, padding=1), 
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, stride=1, padding=1),
            )
            self.relu=nn.ReLU()

    def forward(self, x, watp):
        watp_prod=self.conv(watp)
        watp_sum=self.conv(watp)
        x=torch.mul(x,watp_prod)
        x=torch.add(x,watp_sum)
        x=self.relu(x)
        return x