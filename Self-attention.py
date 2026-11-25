import torch   # I have imported the required libraries to be used for tensor and mathematical computations.
import torch.nn as nn
import torch.nn.functional as F
n_emd = 32     # the dimentions for embedding of tokens
block_size = 8     #the block size used in my project for training
dropout = 0.1      # the rate I used in training
batch_size = 4     # the batch size used in my project for training
class Head(nn.Module):                                   
                    

    def __init__(self, head_size):                      # it is run once when created where head_size = numbers per q, k, v
        super().__init__()             #we use this to switch to our nn.module                
        self.key = nn.Linear(n_emd, head_size, bias=False)     # it makes key which says what am i, like what I have
        self.query = nn.Linear(n_emd, head_size, bias=False)   # it makes the query which tells us what is it looking for
        self.value = nn.Linear(n_emd, head_size, bias=False)   # it makes the value that tells us what it will give us

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # tril is a triangular matrix of 1s and 0s, and it has 65*65 dimention
                                                         

        self.dropout = nn.Dropout(dropout)               # gives 0 to some weights in training randomly

    def forward(self, x):                                
        B,T,C = x.shape                                  # batch, its position, and n_emd
        k = self.key(x)                                  # keys for every character
        q = self.query(x)                                # queries for every character
        
        wei = q @ k.transpose(-2,-1) * C**-0.5           # match each query with each key, then scale
                                                         # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
               
        wei = F.softmax(wei, dim=-1)                     # turn each row of weights into percentages
        wei = self.dropout(wei)                          # dropout on the attention percentages
        
        v = self.value(x)                                # values for every character
        out = wei @ v                                    # take percentages
                                                         # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out                                       # return result
