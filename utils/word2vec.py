import os
import torch
from torch import nn
import pandas as pd
from skimage import io, transform
import datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import re
import nltk
from collections import Counter


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
from scipy.spatial import distance


    
    
class Word2VecDataset(Dataset):
    """
    Takes a HuggingFace dataset as an input, to be used for a Word2Vec dataloader.
    """
    def __init__(self, dataset=None, vocab_size=None, wsize=3):
        self.dataset = dataset
        self.dataset_ = dataset
        self.vocab_size = vocab_size
        self.data = [i for s in self.dataset['moving_window'] for i in s]
        self.dataloader = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # the cat ran --> v1 = [0, 1, 2] # troll 1
        # the dog ran --> v2 = [0, 3, 2] # troll 2
        #  cos(phi) = v1 dot v2
        # need some kind of similarity
        # categories if they share no similar words v1 and v2 should be orthogonal to each other
        # z.sise --> N x latent_dim
        # l.size --> N x k
        # z x l --> (N x l)^T dot (N x k)--> (l x k)
        # new_vec = concat(z, l) --> N x (l + k)
        
        # z = emb(token); classifier(z) --> nn.Linear(latent_dim, k)
        self.embed = nn.Embedding(vocab_size, embedding_size) # vocab.size() --> N x #uniquewords; z.size() --> N x latent_dim 
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        # Encode input to lower-dimensional representation
        logits = self.embed(input)
        
        # Expand hidden layer to predictions
        # logits = self.expand(logits)
    
        return logits
    
class Word2VecObject:
    
    def __init__(self, path=None, device=None, pars=None):
        self.device = device
        self.ss = SnowballStemmer('english')
        self.sw = stopwords.words('english')
        self.model = None
        self.pars = pars
        
        self.dataloader = None
        
        if self.pars.vanilla:
            path = None
        
        if path is None:
            self.dataset = datasets.load_dataset('tweets_hate_speech_detection')
        else:
            self.dataset = self.makeDataset(path)
            
        self.dataset = self.dataset.map(self.split_tokens)
        self.n_v, self.id2tok, self.tok2id, self.vocab = self.vocabSize(dataset=self.dataset)
        self.dataset = self.dataset.map(self.remove_rare_tokens)
        self.dataset = self.dataset.map(self.windowizer)
        print(self.dataset.keys())
        
    def makeDataset(self, path):
        dataset = datasets.load_dataset('json', data_files= path)
        return dataset
        
    
    def split_tokens(self, row):
        
        if (self.pars.train or self.pars.test) and not self.pars.vanilla:
            tmp = []
            for key in row.keys():
                for i in re.split(r" +", re.sub(r'http\S+','',row[key])):
                    if (i not in self.sw) and len(i):
                        # row['all_tokens'].append(self.ss.stem(i))
                        tmp.append(self.ss.stem(i))
            
            row['all_tokens'] = tmp  
             
        else:                  
            row['all_tokens'] = [
                self.ss.stem(i) for i in re.split(
                    r" +", 
                    re.sub(r"[^a-z@# ]", "", row['tweet'].lower())
                    ) if (i not in self.sw) and len(i)
                ]  
            
        return row
    
    def vocabSize(self, dataset=None):
        if (self.pars.train or self.pars.test) and not self.pars.vanilla:
            counts = Counter([i for s in dataset['train']['all_tokens'][0] for i in s])
        else:
            counts = Counter([i for s in dataset['train']['all_tokens'] for i in s])
            
        counts = {k:v for k, v in counts.items() if v>10} # Filtering
        vocab = list(counts.keys())
        n_v = len(vocab)
        id2tok = dict(enumerate(vocab))
        tok2id = {token: id for id, token in id2tok.items()}
        return n_v, id2tok, tok2id, vocab
    
    
    # Now correct tokens
    def remove_rare_tokens(self, row):
        row['tokens'] = [t for t in row['all_tokens'] if t in self.vocab]
        return row
    
    def windowizer(self, row, wsize=3):
        """
        Windowizer function for Word2Vec. Converts sentence to sliding-window
        pairs.
        """
        doc = row['tokens']
        # print(doc)
        # assert False
        wsize = 5 # change from 3
        out = []
        for i, wd in enumerate(doc):
            target = self.tok2id[wd]
            window = [i+j for j in
                    range(-wsize, wsize+1, 1)
                    if (i+j>=0) &
                        (i+j<len(doc)) &
                        (j!=0)]

            out+=[(target, self.tok2id[doc[w]]) for w in window]
        row['moving_window'] = out
        
        return row
    
    def get_distance_matrix(self, wordvecs, metric):
        dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
        return dist_matrix

    def get_k_similar_words(self, word, dist_matrix, k=10):
        idx = self.tok2id[word]
        dists = dist_matrix[idx]
        ind = np.argpartition(dists, k)[:k+1]
        ind = ind[np.argsort(dists[ind])][1:]
        out = [(i, self.id2tok[i], dists[i]) for i in ind]
        return out
    
    def trainWord2Vec(self, latent_dim=None, tweet_mapping=None):
        BATCH_SIZE = self.pars.max
        N_LOADER_PROCS = 10

        self.dataloader = {}        
    
   
        for key in self.dataset.keys():
            self.dataloader.update({key: DataLoader(Word2VecDataset(self.dataset[key], vocab_size=self.n_v),
                                        batch_size=self.pars.max,
                                        shuffle=True,
                                        num_workers=N_LOADER_PROCS,
                                        drop_last=True)})
            
        # Instantiate the model
        EMBED_SIZE = latent_dim # Quite small, just for the tutorial
        self.model = Word2Vec(self.n_v, EMBED_SIZE)

        # Relevant if you have a GPU:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print("USING CUDA? -- >", device)

        # Define training parameters
        LR = 3e-4
        EPOCHS = 10000
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        progress_bar = tqdm(range(EPOCHS * len(self.dataloader['train'])))
        running_loss = []
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for center, context in self.dataloader['train']:

                center, context = center.to(device), context.to(device)
                optimizer.zero_grad()
                logits = self.model(input=context)
                loss = loss_fn(logits, center)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
            epoch_loss /= len(self.dataloader['train'])
            running_loss.append(epoch_loss)
            
        # wordvecs = self.model.expand.weight.cpu().detach().numpy()
        # tokens = ['good', 'father', 'school', 'hate']
        
        # dmat = self.get_distance_matrix(wordvecs, 'cosine')
        # for word in tokens:
        #     print(word, [t[1] for t in self.get_k_similar_words(word, dmat)], "\n")
            
        
            