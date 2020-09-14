#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# In[2]:


cols = ['userid', 'itemid', 'rating', 'timestamp']
root = r'C:\Users\Guan-Ting Chen\Desktop\ml-100k'
train_data = pd.read_csv(os.path.join(root, 'u1.base'), sep='\t', names=cols).drop(columns=['timestamp']).astype(int)
test_data = pd.read_csv(os.path.join(root, 'u1.test'), sep='\t', names=cols).drop(columns=['timestamp']).astype(int)


# In[3]:


n_user, n_item = train_data[['userid', 'itemid']].max()


# In[4]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[5]:


class MovieLens(Dataset):
    def __init__(self, df):
        self.df_values = df.values
        
    def __getitem__(self, idx):
        user, item, rating = self.df_values[idx]
        
        return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([rating])
    
    def __len__(self):
        
        return len(self.df_values)


# In[6]:


class Matrixfactorization(nn.Module):
    def __init__(self, n_user, n_item, mu, dim=20):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.init_embedding()
        self.mu = mu
        
    def init_embedding(self):
        
        self.user_latent.weight.data.normal_(0, 0.02)
        self.item_latent.weight.data.normal_(0, 0.02)
        self.user_bias.weight.data.normal_(0, 1)
        self.item_bias.weight.data.normal_(0, 1)
        
        return self
          
    def forward(self, users, items):
        # indexes of user and items start at 1
        # python start at 1

        u_latent = self.user_latent(users-1).squeeze()
        i_latent = self.item_latent(items-1).squeeze()
        u_bias = self.user_bias(users-1).squeeze()
        i_bias = self.item_bias(items-1).squeeze()
        
        ratings = u_latent.mm(i_latent.transpose(1,0)).diag() + u_bias + i_bias + self.mu

        return ratings


# In[7]:


BATCH_SIZE = 200
dim = 100
LEARNING_RATE = 0.005
N_EPOCHES = 20
WEIGHT_DECAY = 0.002

# dataset
train_loader = DataLoader(MovieLens(train_data),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                    
                          num_workers=0)
test_loader = DataLoader(MovieLens(test_data),
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0)

mu = train_data['rating'].mean()
# devic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print('Current device:', device)

# model
model = Matrixfactorization(n_user, n_item, mu=mu, dim=dim).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# In[8]:


for epo in tqdm(range(N_EPOCHES)):
    training_loss = 0
    for users, items, ratings in train_loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        pre_ratings = model(users, items)
        loss = criterion(pre_ratings, ratings.squeeze())
        training_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    with torch.no_grad() :
        
        training_loss = 0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            pre_ratings = model(users, items)
            loss = criterion(pre_ratings, ratings.squeeze())
            training_loss += loss.item()

        testing_loss = 0
        for users, items, ratings in test_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            pre_ratings = model(users, items)
            loss = criterion(pre_ratings, ratings.squeeze())
            testing_loss += loss.item()
    
    training_rmse = np.sqrt(training_loss / train_loader.dataset.__len__())
    testing_rmse = np.sqrt(testing_loss / test_loader.dataset.__len__())
        
    print('Epoch: {0:2d} / {1}, Traning RMSE: {2:.4f}, Testing RMSE: {3:.4f}'.format(epo+1, 
                                                                                     N_EPOCHES,
                                                                                     training_rmse,
                                                                                     testing_rmse))


# ## Learning map 
# #### Nature Language Procesing
#    - POS Tagging
#    - Parsing
#    - Name Entity Recognition
#    - Semantic Classification
#    - Next Sequuence Predict
#    - Popular Models: Seq2Seq $\Rightarrow$ Attention $\Rightarrow$ Self-Attention $\Rightarrow$ Transformer $\Rightarrow$ BERT $\Rightarrow$  ELECTRA, RoBERTa, XLNet, T5, Reformer
# ---
# #### Computer Vision
#    - Image Processing (Augmentation):
#       - Image Enhancement 
#       - Image Restoration
#    - Object Classification   
#    - Object Detection: 
#       - R-CNN $\Rightarrow$  Fast R-CNN $\Rightarrow$  Faster R-CNN $\Rightarrow$  Mask R-CNN $\Rightarrow$  YOLO v1-v4   
#          - https://medium.com/cubo-ai/%E7%89%A9%E9%AB%94%E5%81%B5%E6%B8%AC-object-detection-740096ec4540
#       - Semantic Segmentation (Pixel-wise)
#          - https://www.topbots.com/semantic-segmentation-guide/#:~:text=Semantic%20Segmentation%20is%20the%20process,every%20pixel%20in%20the%20image.&text=Semantic%20segmentation%20treats%20multiple%20objects,individual%20objects%20(or%20instances).
#       - Instance Segmentation (Pixel-wise)
#          - https://datascience.stackexchange.com/questions/52015/what-is-the-difference-between-semantic-segmentation-object-detection-and-insta
#    - Popular Models: LeNet $\Rightarrow$  AlexNet $\Rightarrow$  VGGNet $\Rightarrow$  Inception(GoogLeNet) v1-v4 $\Rightarrow$  ResNet 
#       - https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306
# ---
# #### Recommender System
#    - Rating Prediction
#    - Sequential Recommendation  
# ---
# #### Generative Aversarial Network
#    - Basic:
#       -  https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-generative-adversarial-network-gan-%E7%94%9F%E6%88%90%E5%B0%8D%E6%8A%97%E7%B6%B2%E8%B7%AF-c672125af9e6
#    - GAN $\Rightarrow$  CGAN $\Rightarrow$  DCGAN $\Rightarrow$  ...
# ---
# #### Reinforcement Learning
#    - Basic:
#       - https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-reinforcement-learning-%E5%BC%B7%E5%8C%96%E5%AD%B8%E7%BF%92-dqn-%E5%AF%A6%E4%BD%9Catari-game-7f9185f833b0
#       - https://www.freecodecamp.org/news/a-brief-introduction-to-reinforcement-learning-7799af5840db/
#    - Deep Q-Network (DQN)
#    - Actor-Critic (A2C, A3C)
# ---
# #### Graph Neural Network
#    - https://medium.com/@z8663z/%E9%96%B1%E8%AE%80%E7%AD%86%E8%A8%98-a-comprehensive-survey-on-graph-neural-networks-%E7%B7%A8%E8%BC%AF%E4%B8%AD-78118deae743
#    - https://medium.com/dair-ai/an-illustrated-guide-to-graph-neural-networks-d5564a551783
#    - https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3
#    
# #### Recommended bloggers:
#    - https://medium.com/@jackshiba
#    - https://medium.com/@super135799
#    - https://medium.com/@chih.sheng.huang821
#    - https://leemeng.tw/index.html#projects
#    - https://taweihuang.hpd.io/category/data-science/
#    - https://medium.com/@chaturangarajapakshe
#    - https://ruder.io/author/sebastian/index.html
#       - https://ruder.io/optimizing-gradient-descent/
# ---
