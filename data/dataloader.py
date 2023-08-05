import pickle
import torch
import os
import numpy as np
import pandas as pd
import logging

from torch.utils.data import Dataset

from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


class Twibot22SampleDataset(Dataset):
    
    def __init__(self, set_name: str = "train"):
        
        data_path = os.path.join("data", "Twibot-22", set_name)
        
        # Define file names
        user_file = "user.json"
        label_file = "label.csv"
        edge_file = "tweet_edges.csv"
        tweet_files = [f"tweet_{i}.csv" for i in range(9)]
        
        print("Reading files...")
        self.user = pd.read_json(
            os.path.join(data_path, user_file),
            orient="index"
        )
        self.label = pd.read_csv(os.path.join(data_path, label_file))
        self.edge = pd.read_csv(os.path.join(data_path, edge_file))
        self.tweet = pd.concat([pd.read_csv(os.path.join(data_path, tweet_file), lineterminator='\n', nrows=200000) for tweet_file in tweet_files])
        print(len(self.tweet))
        
        
        print("Processing user data...")
        self.feature_user()
        
        print("Generating tweet closure graph...")
        self.group_user_tweets_and_tweet_closure_graph()
        
        print("Cleaning resources...")
        del(self.tweet)
        del(self.edge)
        
        
    def feature_user(self):
        self.ids = self.user.pop("id").tolist()
        self.user = self.user.astype("float").fillna(0.0)
    
    
    def process_closure_graph(self, user_tweets):
        # Filtering user's tweets
        user_tweets["from_user"] = (user_tweets["relation"] == "post").astype("int")

        tweet_indices = {tweet_id: i for i, tweet_id in enumerate(user_tweets["target_id"])}

        edge_cp = self.edge[self.edge["source_id"].isin(tweet_indices)].copy(deep=True)
        edge_cp["source_index"] = edge_cp["source_id"].map(tweet_indices)
        edge_cp["target_index"] = edge_cp["target_id"].map(tweet_indices)
        edge_cp.dropna(subset=["source_index", "target_index"], inplace=True)
        edge_cp["source_index"] = edge_cp["source_index"].astype('int')
        edge_cp["target_index"] = edge_cp["target_index"].astype('int')

        tweet_adj = np.array(list(zip(edge_cp["source_index"], edge_cp["target_index"]))).reshape(-1, 2)

        tweet_texts = user_tweets["text"].values
        owned = user_tweets["from_user"].values
        tlabel = user_tweets["label"].values

        return (tweet_adj, tweet_texts, owned, tlabel)
    
    def group_user_tweets_and_tweet_closure_graph(self):
        result = self.tweet.groupby("source_id").parallel_apply(self.process_closure_graph)
        result = result.reindex(index=self.ids)
        self.user_tweets = result
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index):
        user = self.user.iloc[index].values
        user_tweet = self.user_tweets.iloc[index]
        if pd.isna(user_tweet):
            tweet_adj, tweet_texts, owned, tlabel = [
                np.zeros((0, 2)),
                np.zeros((0,)),
                np.zeros((0,)),
                np.zeros((0,))
            ]
        else:
            tweet_adj, tweet_texts, owned, tlabel = self.user_tweets.iloc[index]
        label = self.label.iloc[index]["label"]
        return user, tweet_texts, tweet_adj, owned, label, tlabel
    
    
if __name__ == "__main__":
    dataset = Twibot22SampleDataset(set_name="train")
    torch.save(dataset, "dataset/twibot_22_full_train.pt")
    # dataset = torch.load("dataset/twibot_22_full_train.pt")
    print('Dataset length: ', len(dataset))
    indices = range(700000)
    for idx in tqdm(indices):
        sample_user = dataset[idx]
        if len(sample_user[2]) > 0:
            print('User')
            print(sample_user[0])
            print(sample_user[0].shape)
            print('Encoded tweet (sparse matrix)')
            print(sample_user[1])
            print(sample_user[1].shape)
            print('Tweet adjacency')
            print(sample_user[2])
            print(sample_user[2].shape)
            print('User-post matrix')
            print(sample_user[3])
            print(sample_user[3].shape)
            print('Label')
            print(sample_user[4])
            print(sample_user[4].shape)
            print('Tweet label')
            print(sample_user[5])
            print(sample_user[5].shape)
            print('-' * 50)