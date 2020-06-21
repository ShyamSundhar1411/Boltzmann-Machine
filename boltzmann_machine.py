#Importing Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
#Importing Dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = "::", header = None, engine = 'python',encoding = 'Latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',header = None, engine = 'python', encoding = 'Latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = "::",header = None, engine = 'python', encoding = 'Latin-1')
#Training Set and Test Set
train_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
train_set = np.array(train_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')
#Getting Numbers of users and movies
user_no = int(max(max(train_set[:,0]), max(test_set[:,0])))
movie_no = int(max(max(train_set[:,1]),max(test_set[:,1])))
#Converting Data into array
def con(d):
    new_data = []
    for i in range(1,user_no+1):
        movid = d[:,1][d[:,0] == i]
        rat = d[:,2][d[:,0] == i]
        ratings = np.zeros(movie_no)
        ratings[movid - 1] = rat
        new_data.append(list(ratings))
    return new_data
train_set = con(train_set)
test_set = con(test_set)
#Connecting with Torch Tensors
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)
#Converting ratings to binary
train_set[train_set == 0] = -1
train_set[train_set == 1] = 0
train_set[train_set == 2] = 0
train_set[train_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
#Restricted Boltzmann Machine
class RBM():
    def __init__(self, nv,nh):
        self.w = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sam_h(self,x):
        p = torch.mm(x, self.w.t())
        acti = p + self.a.expand_as(p)
        prob = torch.sigmoid(acti)
        return prob,torch.bernoulli(prob)
    def sam_v(self,y):
        p = torch.mm(y, self.w)
        acti = p+self.b.expand_as(p)
        prob = torch.sigmoid(acti)
        return prob, torch.bernoulli(prob)
    def train(self,v,k,ph,phk):
        self.w += (torch.mm(v.t(),ph) - torch.mm(k.t(),phk)).t()
        self.b += torch.sum((v-k),0)
        self.a += torch.sum((ph-phk),0)
nv = len(train_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)
#Training
epochs = 10
for i in range(1,epochs+1):
    loss = 0
    s = 0.
    for j in range(0,user_no - batch_size,batch_size):
        v = train_set[j:j+batch_size]
        vk =train_set[j:j+batch_size]
        ph,_ = rbm.sam_h(v)
        for k in range(10):
            _,hk = rbm.sam_h(vk)
            _,vk = rbm.sam_v(hk)
            vk[v<0] = v[v<0]
        phk,_ = rbm.sam_h(vk)
        rbm.train(v,vk,ph,phk)
        loss += torch.mean(torch.abs(v[v>=0]-vk[v>=0]))
        s+=1
    print('Epoch :'+str(i)+' Loss :'+str(loss/s))
#Testing
loss = 0
s = 0.
for j in range(user_no):
    v = train_set[j:j+1]
    vt = test_set[j:j+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sam_h(v)
        _,v = rbm.sam_v(h)
        loss += torch.mean(torch.abs(vt[vt>=0]- v[vt>=0]))
        s+=1.
print('Test Loss :'+str(loss/s))

        


        
