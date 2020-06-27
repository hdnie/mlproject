import torch as torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
class lstm(nn.Module):
    def __init__(self):
        super(lstm,self).__init__()
        self.LSTM=nn.LSTM(400,128,batch_first=True,num_layers=3)
        self.output=nn.Linear(128,4)
    def forward(self,x):
        x,(h_n,c_n)=self.LSTM(x)
        x=self.output(x)
        x=nn.Softmax()(x)
        return x
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.l1=nn.Linear(400,128)
        self.l2=nn.Linear(128,4)
    def forward(self,x):
        x=self.l1(x)
        x=nn.ReLU(inplace=True)(x)
        x=self.l2(x)
        x=nn.Softmax()(x)
        return x
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):

        # -1 把第一个维度保持住
        return x.reshape((x.shape[0],-1))
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cv1=nn.Conv2d(1,4,10)
        self.max1=nn.MaxPool2d(3,stride=2)
        self.cv2=nn.Conv2d(4,4,5,padding=3)
        self.max2=nn.MaxPool2d(3,stride=2)
        self.cv3=nn.Sequential(nn.Conv2d(4,8,3,padding=2),nn.ReLU(inplace=True),
                               nn.Conv2d(8,8,3,padding=2),nn.ReLU(inplace=True),
                               nn.Conv2d(8,8,3,padding=2),nn.ReLU(inplace=True))
        self.max3=nn.MaxPool2d(3,stride=2)
        self.flt=Flatten()
        self.l1=nn.Linear(128,64)
        self.l2=nn.Linear(64,4)
    def forward(self, x):
        x=self.cv1(x)
        x=nn.ReLU(inplace=True)(x)
        x=self.max1(x)
        x=self.cv2(x)
        x=nn.ReLU(inplace=True)(x)
        x=self.max2(x)
        x=self.cv3(x)
        x=self.max3(x)
        x=self.flt(x)
        x=self.l1(x)
        x=nn.ReLU(inplace=True)(x)
        x=self.l2(x)
        x=nn.Softmax()(x)
        return x
class utime(nn.Module):
    def __init__(self,classnum=4,samplerate=400,sec=1):
        super(utime,self).__init__()
        self.conv1=nn.Conv1d(1,16,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn1=nn.BatchNorm1d(16)
        self.conv2=nn.Conv1d(16,16,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn2=nn.BatchNorm1d(16)
        self.maxpl1=nn.MaxPool1d(10)
        self.conv3=nn.Conv1d(16,32,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn3=nn.BatchNorm1d(32)
        self.conv4=nn.Conv1d(32,32,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn4=nn.BatchNorm1d(32)
        self.maxpl2=nn.MaxPool1d(8)
        self.conv5=nn.Conv1d(32,64,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn5=nn.BatchNorm1d(64)
        self.conv6=nn.Conv1d(64,64,5,padding=4,dilation=2,padding_mode="zeros")
        self.bn6=nn.BatchNorm1d(64)
        self.maxpl3=nn.MaxPool1d(6)
        self.conv7=nn.Conv1d(64, 128, 5, padding=4,dilation=2, padding_mode="zeros")
        self.bn7=nn.BatchNorm1d(128)
        self.conv8=nn.Conv1d(128, 128, 5, padding=4,dilation=2, padding_mode="zeros")
        self.bn8=nn.BatchNorm1d(128)
        self.maxpl4=nn.MaxPool1d(4)
        self.deconv1=nn.Conv1d(128, 256, 5, padding=4,dilation=2, padding_mode="zeros")
        self.dbn1=nn.BatchNorm1d(256)
        self.deconv2=nn.Conv1d(256, 256, 5, padding=4,dilation=2, padding_mode="zeros")
        self.dbn2=nn.BatchNorm1d(256)
        self.upsamp1=nn.Upsample(scale_factor=4,mode="nearest")

        self.deconv3=nn.Conv1d(256, 128, 4, padding=1,dilation=1, padding_mode="zeros")
        self.dbn3=nn.BatchNorm1d(128)
        self.deconv4 = nn.Conv1d(256, 128, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn4 = nn.BatchNorm1d(128)
        self.deconv5 = nn.Conv1d(128, 128, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn5 = nn.BatchNorm1d(128)
        self.upsamp2=nn.Upsample(scale_factor=6,mode="nearest")

        self.deconv6 = nn.Conv1d(128, 64, 6, padding=2, dilation=1, padding_mode="zeros")
        self.dbn6 = nn.BatchNorm1d(64)
        self.deconv7 = nn.Conv1d(128, 64, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn7 = nn.BatchNorm1d(64)
        self.deconv8 = nn.Conv1d(64, 64, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn8 = nn.BatchNorm1d(64)
        self.upsamp3 = nn.Upsample(scale_factor=8, mode="nearest")

        self.deconv9 = nn.Conv1d(64, 32, 8, padding=3, dilation=1, padding_mode="zeros")
        self.dbn9 = nn.BatchNorm1d(32)
        self.deconv10 = nn.Conv1d(64, 32, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn10 = nn.BatchNorm1d(32)
        self.deconv11 = nn.Conv1d(32, 32, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn11 = nn.BatchNorm1d(32)
        self.upsamp4 = nn.Upsample(scale_factor=10, mode="nearest")

        self.deconv12 = nn.Conv1d(32, 16, 10, padding=4, dilation=1, padding_mode="zeros")
        self.dbn12 = nn.BatchNorm1d(16)
        self.deconv13 = nn.Conv1d(32, 16, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn13 = nn.BatchNorm1d(16)
        self.deconv14 = nn.Conv1d(16, 16, 5, padding=2, dilation=1, padding_mode="zeros")
        self.dbn14 = nn.BatchNorm1d(16)

        self.deconv15=nn.Conv1d(16,classnum,1,padding=0,dilation=1)
        self.avepool1=nn.AvgPool1d(3000)
        self.deconv16=nn.Conv1d(classnum,classnum,1,padding=0,dilation=1)

        self.sec=sec
        self.samplerate=samplerate

    def forward(self,x):
        e1=self.conv1(x)
        e1=F.relu(e1)
        '''print("e1:",e1.shape)'''
        eb1=self.bn1(e1)
        e2=self.conv2(eb1)
        e2=F.relu(e2)
        '''print("e2:",e2.shape)'''
        eb2=self.bn2(e2)
        maxpool1=self.maxpl1(eb2)
        '''print("maxpool1:", maxpool1.shape)'''

        e3=self.conv3(maxpool1)
        e3=F.relu(e3)
        '''print("e3:", e3.shape)'''
        eb3=self.bn3(e3)
        e4=self.conv4(eb3)
        e4=F.relu(e4)
        '''print("e4:", e4.shape)'''
        eb4=self.bn4(e4)
        maxpool2=self.maxpl2(eb4)
        '''print("maxpool2:", maxpool2.shape)'''
        e5=self.conv5(maxpool2)
        e5=F.relu(e5)
        '''print("e5:", e5.shape)'''
        eb5=self.bn5(e5)
        e6=self.conv6(eb5)
        e6=F.relu(e6)
        '''print("e6:", e6.shape)'''
        eb6=self.bn6(e6)


        usamp3 = self.upsamp3(eb6)
        '''print("usamp3:", usamp3.shape)'''

        usamp3 = F.pad(usamp3, [1, 0])
        de9 = self.deconv9(usamp3)
        de9 = F.relu(de9)
        '''print("de9:", de9.shape)'''
        deb9 = self.dbn9(de9)
        cropnum = eb4.shape[2] - deb9.shape[2]

        if (cropnum == 0):
            deb9 = torch.cat([deb9, eb4], 1)
        else:
            deb9 = torch.cat([deb9, eb4[:, :, int(cropnum / 2):-int((cropnum + 1) // 2)]], 1)
        de10 = self.deconv10(deb9)
        de10 = F.relu(de10)
        '''print("de10:", de10.shape)'''
        deb10 = self.dbn10(de10)
        de11 = self.deconv11(deb10)
        de11 = F.relu(de11)
        '''print("de11:", de11.shape)'''
        deb11 = self.dbn11(de11)
        usamp4 = self.upsamp4(deb11)
        '''print("usamp4:", usamp4.shape)'''

        usamp4 = F.pad(usamp4, [1, 0])
        de12=self.deconv12(usamp4)
        de12=F.relu(de12)
        '''print("de12:", de12.shape)'''
        deb12=self.dbn12(de12)
        cropnum = eb2.shape[2] - deb12.shape[2]
        if (cropnum == 0):
            deb12 = torch.cat([deb12, eb2], 1)
        else:
            deb12 = torch.cat([deb12, eb2[:, :, int(cropnum / 2):-int((cropnum + 1) // 2)]], 1)
        de13=self.deconv13(deb12)
        de13=F.relu(de13)
        '''print("de13:", de13.shape)'''
        deb13=self.dbn13(de13)
        de14=self.deconv14(deb13)
        de14=F.relu(de14)
        '''print("de14:", de14.shape)'''
        deb14=self.dbn14(de14)

        de15=self.deconv15(deb14)
        de15=torch.tanh(de15)
        '''print("de15:", de15.shape)'''
        padnum=x.shape[2]-de15.shape[2]
        de16=F.pad(de15,[(padnum//2),((padnum+1)//2)])
        '''print("de16:", de16.shape)'''
        '''reshape16=torch.reshape(de16,[x.shape[0],5,35,3000])'''
        '''print("reshape16:", reshape16.shape)'''
        avepool=nn.AvgPool1d(self.sec*self.samplerate)(de16)
        '''print("avepool:", avepool.shape)'''
        res=self.deconv16(avepool)
        res=F.softmax(res,dim=1)
        res=res.squeeze()
        '''print("res:",res.shape)'''
        return res
'''import numpy as np
model=lstm()
a=np.zeros((2,1,400))
a=torch.tensor(a,dtype=torch.float)
b=model(a)
print(b.shape)'''