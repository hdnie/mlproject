import xlrd
import numpy as np
import torch.utils.data as Data
import torch
import scipy.io as scio
def normal(a):
    return (a-np.mean(a))/(np.var(a))
def get_fold(npy,m,n,dim,device):
    num=npy.shape[0]
    for i in range(1200):
        npy[i,:-1]=normal(npy[i,:-1])
    cvnum=num//m
    cvdata=npy[n*cvnum:(n+1)*cvnum,:]
    trdata1=npy[:n*cvnum,:]
    trdata2=npy[(n+1)*cvnum:,:]
    trdata = np.concatenate((trdata1, trdata2), axis=0)
    '''for i in range(44):
        print(np.max(trdata[:,i]))'''
    train=torch.Tensor(trdata[:,:-1]).to(device)
    trlabel=torch.Tensor(trdata[:,-1]).to(device)
    cv=torch.Tensor(cvdata[:,:-1]).to(device)
    cvlabel=torch.Tensor(cvdata[:,-1]).to(device)
    if(dim==2):
        trshape=train.shape
        cvshape=cv.shape
        train=train.reshape((trshape[0],20,20))
        cv=cv.reshape((cvshape[0],20,20))

    trdata=Data.TensorDataset(train,trlabel.int())
    cvdata=Data.TensorDataset(cv,cvlabel.int())
    return trdata,cvdata
'''def getalldata():
    alldata = np.zeros((1200, 401))
    for i in range(4):
        name = str(i) + ".mat"
        data = scio.loadmat(name)
        data = data[list(data.keys())[3]]
        for j in range(300):
            alldata[300 * i + j][:400] = normal(data[j * 400:(j + 1) * 400, 0])
            alldata[300 * i + j][400] = int(i)
    np.random.shuffle(alldata)
    return alldata
def mat2npy2d(device):
    trdata=np.zeros((800,401))
    tedata = np.zeros((400, 401))
    for i in range(4):
        name=str(i)+".mat"
        data=scio.loadmat(name)
        data=data[list(data.keys())[3]]
        for j in range(200):
            trdata[200*i+j][:400]=data[j*400:(j+1)*400,0]
            trdata[200 * i + j][400]=int(i)
        for k in range(100):
            tedata[100*i+k,:400]=data[80000+k*400:80000+(k+1)*400,0]
            tedata[100*i+k,400]=int(i)
    for i in range(800):
        trdata[i,:-1]=normal(trdata[i,:-1])
    for i in range(400):
        tedata[i, :-1] = normal(tedata[i, :-1])

    trdata=torch.Tensor(trdata).to(device)
    tedata=torch.Tensor(tedata).to(device)
    trlabel=trdata[:,-1]
    telabel=tedata[:,-1]
    trdata=trdata[:,:-1].reshape((800,20,20))
    tedata=tedata[:,:-1].reshape((400,20,20))

    trdata=Data.TensorDataset(trdata,trlabel.int())
    tedata=Data.TensorDataset(tedata,telabel.int())
    return trdata,tedata'''
'''npy=np.load("alldata.npy")
a,b=get_fold(npy,5,0,dim=2,device="cuda:0")'''

