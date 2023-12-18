import os
import argparse 
import pandas as pd 
import numpy as np
import torch
import networkx as nx
from utils.word2vec import *
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx, from_networkx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if not os.path.exists("data/csv/"):
    os.makedirs("data/csv/")
    
if not os.path.exists("data/processed/"):
    os.makedirs("data/processed/")
    
if not os.path.exists("data/raw/"):
    os.makedirs("data/raw/")

if not os.path.exists("data/imgs/"):
    os.makedirs("data/imgs")
    

tweets_path = "data/raw/" + "tweets.csv"

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', type=boolean_string, default='False', help='train Word2Vec embedder')
parser.add_argument('--test', type=boolean_string, default='False', help='test Word2Vec embedder')
parser.add_argument('--vanilla', type=boolean_string, default='False', help='test Word2Vec embedder')
parser.add_argument('--all', type=boolean_string, default='False', help='go for broke on the entire dataset')
parser.add_argument('--max', type=int, default=100, help='max size of training data')
parser.add_argument('--latent_dim', type=int, default=100, help='max size of training data')


args = parser.parse_args()
print(args)

def get_Tweets(path="data/csv/", **kwargs):
    files = [f for f in os.listdir(path)]
    
    tweet_list = []
    
    want = [
        'external_author_id',
        'content', # need to encode
        'language', # filter for english then drop
        # 'publish_date',
        # 'following',
        # 'followers',
        # 'post_type', # might need to encode
        # 'account_type', #need to encode
        'retweet', # use for IdEncoder
        'account_category', #need to encode
        'tweet_id', 
    ]
    
    for filename in files:
        f = open(path + filename) 
        header, extension = filename.split('.')
    
        df = pd.read_csv(path + filename)
        columns = list(df.columns)            
        drop = [i for i in columns if i not in want] # make list of columns we don't want
        df.drop(columns=drop, inplace=True) # drop columns we don't want
        df.drop(df.loc[df['language']!='English'].index, inplace=True) # Filter language only english
        df.drop(columns=['language'], inplace=True)
        df.drop(df.loc[df['account_category']=='Unknown'].index, inplace=True) # Filter out unknown account_category
        df.drop(df.loc[df['account_category']=='NonEnglish'].index, inplace=True) # Filter out NonEnglish account_category
        tweet_list.append(df)

    
    df_tweets = pd.concat(tweet_list, ignore_index=True)
    print(df_tweets.head())
    df_tweets.to_csv(tweets_path,index=False)


def load_node_csv(path, index_col, max=100, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    df = df.iloc[:max]
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    trolltypes = None
    
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
        
        
        trolltype_encoder = encoders['account_category']
        trolltypes = trolltype_encoder.mapping

    return x, mapping, trolltypes

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, max=100,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    df = df.iloc[:max]
            
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    

    edge_attr = None
    print("EDGE ENCODERS --> ", encoders)
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class ContentEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', path=None, device=None, tweet_mapping=None, **kwargs):
        self.device = device
        self.dataloader = None
        print("MAKIN")
        if args.train:
            index_col='tweet_id'
            key='content'
            df = pd.read_csv(path, index_col=index_col, **kwargs)
            df = df.iloc[:args.max]
            df = df[key]
            df.reset_index(drop=True, inplace=True)
            json = df.to_json()
            
            self.row_names = df.index
            # print(row_names)
            # for row in row_names:
            #     print(type(row))
            #     print(row)
            # assert False
                  
            json_path = "data/json/content.json" 
            
            with open(json_path, "w") as outfile:
                outfile.write(json)
            
            self.model = Word2VecObject(path=json_path, device=self.device, pars=args)
            self.model.trainWord2Vec(latent_dim=args.latent_dim, tweet_mapping=self.row_names)
            self.dataloader = self.model.dataloader

        else:    
            self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        if (args.train and not args.test) and not args.vanilla:
            logits_list = []
            for center, context in self.dataloader['train']:
                center, context = center.to(self.device), context.to(self.device)
                logits = self.model.model(input=context)
                logits_list.append(logits)
            x = torch.cat(logits_list, axis=1)
            print("x.size() --> ", x.size())
            args.train = False
            return x.cpu()
            
        if (args.test and not args.train) and not args.vanilla:
            logits_list = []
            for center, context in self.dataloader['test']:
                center, context = center.to(self.device), context.to(self.device)
                logits = self.model.model(input=context)
                logits_list.append(logits)
            x = torch.cat(logits_list, axis=1)
            print(x.size())
            
            return x.cpu()
            
        else:
            x = self.model.encode(df.values, show_progress_bar=True,
                                convert_to_tensor=True, device=self.device)
            return x.cpu()
    
class TrollTypeEncoder:
    def __init__(self, device=None):
        self.device = device
        self.mapping = None
        
    def __call__(self, df):
        
        trolltypes = set(value for value in df)
        self.mapping = {trolltype: i for i, trolltype in enumerate(trolltypes)}
        print("mapping: \n", self.mapping)

        x = torch.zeros(len(df), len(self.mapping))
        
        for i, trolltype in enumerate(df):
            x[i, self.mapping[trolltype]] = 1
        print("x_troll.size(): ", x.size())    
        return x

class RetweetEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def drawGraph(data=None):
    
    G = to_networkx(data, to_undirected=False)
    # print(G.nodes[])
    # assert False
    fig = plt.figure(1, figsize=(11, 17), dpi=860)
    # pos = nx.kamada_kawai_layout(G, dim=2)
    pos = nx.spiral_layout(G, scale=2, center=None, dim=2, resolution=1.0, equidistant=True)
    
    nx.draw_networkx(
        G, 
        pos=pos, 
        node_size=400,
        font_size=4,
        arrowsize=8,
        
    )
    
    plt.savefig("data/imgs/" + str(args) + "Graph.png", format="PNG")
    
def spectral_embedding(z=None, trolltypes=None):
    from sklearn.manifold import SpectralEmbedding
    # Apply spectral embedding
    # print("DOOOD ---> ", z.size(1))
    # assert False
    se = SpectralEmbedding(
        n_components=z.size(1), affinity='nearest_neighbors', n_neighbors=10, random_state=42)
    X_se = se.fit_transform(z)
    
    
    if z.size(0) < 30.0:
        perplexity = 3.0
    else:
        perplexity = 30.0
        
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
    tsne_proj = tsne.fit_transform(X_se)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8), dpi=420)
    print("WORKS")
    print(X_se.shape)
    print(tsne_proj.shape)
    print()
    
    # assert False
      # Plot those points as a scatter plot and label them based on the pred labels
    z_pred = z[:,-len(trolltypes):]
    pred = torch.argmax(z_pred, dim=1)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8), dpi=420)
    
    for lab in range(len(trolltypes.keys())):
        # print("lab --> ", lab)
        indices = pred==lab
        # indices ==lab
        ax.scatter(
            tsne_proj[indices,0],
            tsne_proj[indices,1], 
            c=np.array(cmap(lab)).reshape(1,4), 
            # label = lab,
            label= {i for i in trolltypes if trolltypes[i]==lab},
            alpha=0.5
            )
    ax.legend(fontsize='small', markerscale=1)
    
    plt.savefig("data/imgs/" + "SPECTRAL" + str(args) + "SPECTRALtSNE.png", format="PNG")
    
    
def gen_tSNE(z=None, trolltypes=None):
    print(trolltypes)
    
    if z.size(0) < 30.0:
        perplexity = 3.0
    else:
        perplexity = 30.0
        
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
    tsne_proj = tsne.fit_transform(z)   
    z_pred = z[:,-len(trolltypes):]
    pred = torch.argmax(z_pred, dim=1)
       
    
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8), dpi=420)
    
    for lab in range(len(trolltypes.keys())):
        # print("lab --> ", lab)
        indices = pred==lab
        # indices ==lab
        ax.scatter(
            tsne_proj[indices,0],
            tsne_proj[indices,1], 
            c=np.array(cmap(lab)).reshape(1,4), 
            # label = lab,
            label= {i for i in trolltypes if trolltypes[i]==lab},
            alpha=0.5
            )
    ax.legend(fontsize='small', markerscale=1)
    
    plt.savefig("data/imgs/" + str(args) + "tSNE.png", format="PNG")
    # plt.savefig("data/imgs/" +  "UNTRAINEDtSNE.png", format="PNG")

#####################
# Main
#####################

def main():
    if os.path.isfile(tweets_path):
        print("combined tweets already exist")
    else:
        print("combining tweets")    
        tweets = get_Tweets()
        
    print("head: \n", pd.read_csv(tweets_path).head())
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    
    if args.all:
        args.max = len(pd.read_csv(tweets_path).index)
        print("# all tweets --> ", args.max -1)
        
    else:
        print("# tweets --> ", args.max -1)
        
    max=args.max
    
    tweet_x, tweet_mapping, trolltypes = load_node_csv(
        tweets_path, 
        index_col='tweet_id', 
        encoders={
            'content': ContentEncoder(path=tweets_path, device=device),
            'account_category': TrollTypeEncoder(device=device),
        },
        max=max
    ) # x, mapping, categories
    
    print("tweet_x")
    print(tweet_x.size())
    
    print("len(list(tweet_mapping.keys()))")
    print(len(list(tweet_mapping.keys())))
    
    
    _, user_mapping, __ = load_node_csv(
        tweets_path, 
        index_col='external_author_id', 
        max=max
    )
   
    print("len(list(user_mapping.keys()))")
    print(len(list(user_mapping.keys())))
    
    data = HeteroData()
    

    data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    data['tweet'].x = tweet_x

    edge_index, edge_label = load_edge_csv(
        tweets_path,
        src_index_col='external_author_id',
        src_mapping=user_mapping,
        dst_index_col='tweet_id',
        dst_mapping=tweet_mapping,
        encoders={'retweet': RetweetEncoder(dtype=torch.long)},
        max=max
    )
    
    data['user', 'retweet' ,'tweet'].edge_index = edge_index
    data['user', 'retweet' ,'tweet'].edge_label = edge_label
    
    print(data)
    print(data.validate(raise_on_error=True))
    # assert False
    
    
    drawGraph(data=data)
    # assert False
    spectral_embedding(z=data['tweet'].x, trolltypes=trolltypes)
    gen_tSNE(z=data['tweet'].x, trolltypes=trolltypes)
    print("DONE")
    
    
if __name__ == "__main__":
    main()