# basic package
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


cols = ['userid', 'itemid', 'rating', 'timestamp']
root = r'C:\Users\Guan-Ting Chen\Desktop\ml-100k'
train_data = pd.read_csv(os.path.join(root, 'u1.base'), sep='\t', names=cols).drop(columns=['timestamp']).astype(int)
test_data = pd.read_csv(os.path.join(root, 'u1.test'), sep='\t', names=cols).drop(columns=['timestamp']).astype(int)

n_user, n_item = train_data[['userid', 'itemid']].max()

# pytorch package
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# create dataset
class MovieLens(Dataset):
    def __init__(self, df):
        self.df_values = df.values
        
    def __getitem__(self, idx):
        user, item, rating = self.df_values[idx]
        
        return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([rating])
    
    def __len__(self):
        
        return len(self.df_values)
    
# built model
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



# define hyperparameters
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
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print('Current device:', device)

# model
model = Matrixfactorization(n_user, n_item, mu=mu, dim=dim).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# begin training
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

