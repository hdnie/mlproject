from dataloader import *
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse
import math
from matrix import *
from model import *
import os
#学习率调整
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.5 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr is set to ",lr)
#命令行参数
parser = argparse.ArgumentParser(description="Prepare a data folder for a"
                                                 "CV experiment setup.")
parser.add_argument("--batchsize", type=int, default=3,
                        help="batchsize,default=3")
parser.add_argument("--start_split", type=int, default=0,
                        help="start from which split")
parser.add_argument("--device",type=str,default="cuda:0",help="use which device")
parser.add_argument("--foldnum", type=int, default=5,
                        help="How much folds")
parser.add_argument("--start_epoch", type=int, default=-1,
                        help="Continue train")
parser.add_argument("--lr", type=float, default=5e-5,
                        help="learn rate")
parser.add_argument("--modelpth", type=str, default="",
                        help="pth file dir")
parser.add_argument("--opt", type=str, default="adam",
                        help="use which optim")

parser=parser.parse_args()
batchSize=parser.batchsize
foldnum=parser.foldnum
start_epoch=parser.start_epoch
device=torch.device(parser.device)
opt=parser.opt
learn_rate=parser.lr
print(device," is used ")
classnum=4
ndatas=1
npy=np.load("/home/nhd/mlproject/alldata.npy")
temp="/home/nhd/mlproject/"
optpath=os.path.join(temp,opt+"1")
if (os.path.exists(optpath)==False):
    os.makedirs(optpath)
for cvrank in range(5):
    cvpath=os.path.join(optpath,str(cvrank))
    trainpair,testpair=get_fold(npy,5,cvrank,dim=1,device=device)
    model = utime().to(device)
    trainloder=torch.utils.data.DataLoader(trainpair,batch_size=batchSize,shuffle=True)
    testloader=torch.utils.data.DataLoader(testpair,batch_size=batchSize)

    maxacc=0
    maxrec=0
    maxpre=0
    criterion=nn.CrossEntropyLoss()
    if(opt=="adam"):
        optimizer=optim.Adam(model.parameters(),lr=learn_rate)
    if(opt=="sgd"):
        optimizer=optim.SGD(model.parameters(),lr=learn_rate)
    if(opt=="adagrad"):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate)
    if(opt=="mome"):
        optimizer = optim.SGD(model.parameters(), lr=learn_rate,momentum=0.9)
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(300):
        count=0
        matrix=np.zeros((classnum,classnum),dtype=int)
        running_loss=0.0
        for i,data in enumerate(trainloder,0):
            inputs,labels=data
            inputs=inputs.reshape((-1,1,400))
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            outputs = outputs.reshape((-1, classnum, ndatas))
            labels=labels.reshape((-1,1)).long()

            loss=criterion(outputs,labels)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            count+=1

            for j in range(outputs.shape[0]):
                for k in range(outputs.shape[2]):
                    outlabel = torch.argmax(outputs, axis=1)[j][k]
                    realabel = labels[j][0]
                    matrix[outlabel][realabel] += 1
        '''if(epoch%20==0):
            print("cv:",cvrank,"epoch:",epoch,"loss:", running_loss.item()/count)
        if (epoch%20 == 0):
            print_confusion_matrix(matrix,"training",classnum)'''

        if epoch>0:
            tmatrix = np.zeros((classnum, classnum), dtype=int)
            bestmatrix = np.zeros((classnum, classnum), dtype=int)

            for testi, testdata in enumerate(testloader, 0):
                testinputs, testlabels = testdata
                testinputs = testinputs.reshape((-1, 1, 400))
                testoutputs = model.forward(testinputs)

                testoutputs = testoutputs.reshape((-1, classnum, ndatas))
                for j in range(testoutputs.shape[0]):
                    for k in range(testoutputs.shape[2]):
                        toutlabel = torch.argmax(testoutputs, axis=1)[j][k]
                        trealabel = testlabels[j]
                        tmatrix[toutlabel][trealabel] += 1
            tacc = get_acc(tmatrix)
            trec=get_rec(tmatrix,classnum)
            tpre=get_pre(tmatrix,classnum)
            fscore=2*tpre*trec/(tpre+trec)
            fscore=round(fscore,4)
            trloss=round(running_loss.item(),4)
            fspath=os.path.join(cvpath,"fs")
            if(os.path.exists(fspath)==False):
                os.makedirs(fspath)
            lspath=os.path.join(cvpath,"ls")
            if (os.path.exists(lspath) == False):
                os.makedirs(lspath)
            np.savetxt(os.path.join(lspath, "epo-" + str(epoch) + "ls-" + str(trloss) + ".txt"), tmatrix, delimiter=',',fmt='%d')
            np.savetxt(os.path.join(fspath,"epo-" + str(epoch) + "fs-" + str(fscore) + ".txt"), tmatrix, delimiter=',',fmt='%d')
            '''if epoch%10==0:
                print("testing....................................................")
                print("CV:",cvrank,"Epoch:",epoch)
                print_confusion_matrix(tmatrix, "testing", classnum)'''

