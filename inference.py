import scipy
from model.SOBOG import SOBOG
import networkx as nx
import pickle
import re
import numpy as np
import pandas as pd
import torch


class Inference:

    def __init__(self):
        model_path = 'model/model_3_0.pt'
        tfidf_path = 'vect/vectorizer_20k.pk'
        user_mean_path = 'ckpts/mean.csv'
        user_std_path = 'ckpts/std.csv'
        self.user_columns = [
                'statuses_count',
                'followers_count',
                'friends_count',
                'favourites_count',
                'listed_count',
                'default_profile',
                'default_profile_image',
                'protected',
                'verified',
                'updated',
                'created_at',
                'name',
                'screen_name',
                'description'
            ]
        model_dict = {
            "n_user_features": 20,
            "d_user_embed": 20,
            "n_post_features": 20000,
            "d_post_embed": 128,
            "n_gat_layers": 3,
            "d_user_cls": 32,
            "n_user_cls_layer": 2,
            "d_post_cls": 64,
            "n_post_cls_layer": 3
        }
        
        self.model = self.load_model(model_path, model_dict)
        self.vectorizer = self.load_vectorizer(tfidf_path)
        self.user_mean = pd.read_csv(user_mean_path).values.T
        self.user_std = pd.read_csv(user_std_path).values.T

    def create_user_dataframe(self, user):
        return pd.DataFrame(
            [user],
            columns=self.user_columns
        )

    def preprocessing_user(self, user_df, user_mean, user_std):
        if 'updated' in user_df.columns:
            age = (
                pd.to_datetime(user_df.loc[:, 'updated']) - 
                pd.to_datetime(user_df.loc[:, 'created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        else:
            age = (
                pd.to_datetime(pd.to_datetime('today')) - 
                pd.to_datetime(user_df.loc[:, 'created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        user_df['tweet_freq'] = user_df['statuses_count'] / age
        user_df['followers_growth_rate'] = user_df['followers_count'] / age
        user_df['friends_growth_rate'] = user_df['friends_count'] / age
        user_df['favourites_growth_rate'] = user_df['favourites_count'] / age
        user_df['listed_growth_rate'] = user_df['listed_count'] / age
        user_df['followers_friends_ratio'] = user_df['followers_count'] / np.maximum(user_df['friends_count'], 1)
        user_df['screen_name_length'] = user_df['screen_name'].str.len()
        user_df['num_digits_in_screen_name'] = user_df['screen_name'].str.count('\d')
        user_df['name_length'] = user_df['name'].str.len()
        user_df['num_digits_in_name'] = user_df['name'].str.count('\d')
        user_df['description_length'] = user_df['description'].str.len()
        user_df = user_df.select_dtypes('number').fillna(0.0)
        print(user_df)
        return (user_df - user_mean) / user_std

    def preprocessing_tweet(self, row: str):
        URL_PATTERN = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        row = bytes(row, 'utf-8').decode('latin-1')
        print(row)
        rowlist = str(row).split()
        rowlist = [word.strip() for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '#') else "hashtagtag" for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '@') else "usertag" for word in rowlist]
        rowlist = [word.lower() for word in rowlist]
        rowlist = [re.sub(URL_PATTERN, "urltag", word) for word in rowlist]
        return " ".join(rowlist)

    def vectorizing_tweet(self, tweets):
        print(tweets)
        v = self.vectorizer.transform(tweets)
        return v.A[np.newaxis, :]

    def load_model(self, path, model_dict):
        model = SOBOG(
            gpu=0,
            **model_dict
        )
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def load_vectorizer(self, path):
        with open(path, 'rb') as fin:
            vect = pickle.load(fin)
        return vect

    def generate_adj_matrix(self, tweet_df):
        graph = nx.from_pandas_edgelist(tweet_df, "parent_id", "id", create_using=nx.DiGraph())
        graph.remove_node(0)
        adj = nx.adjacency_matrix(
            graph,
            nodelist=tweet_df["id"].values
        ).A
        return adj[np.newaxis, :]
    
    def generate_user_post_matrix(self, tweet_df):
        return (tweet_df['parent_id'] == 0).astype('int').values[np.newaxis, :]

    def inference(self, user, tweet, adj, up):
        user = torch.Tensor(user)
        tweet = torch.Tensor(tweet)
        adj = torch.Tensor(adj)
        up = torch.Tensor(up)
        print(user.shape, tweet.shape, adj.shape, up.shape)
        user_pred, tweet_pred = self.model.forward(user, tweet, adj, up)
        return user_pred, tweet_pred

    def predict(self, user_object, tweet_df):
        user_df = self.create_user_dataframe(user_object)
        user_df = self.preprocessing_user(user_df, self.user_mean, self.user_std)
        user = user_df.values
        print(user)
        tweet_df["text"] = tweet_df["text"].apply(self.preprocessing_tweet)
        tweet = self.vectorizing_tweet(tweet_df["text"])
        print(scipy.sparse.csr_matrix(tweet[0]))
        adj = self.generate_adj_matrix(tweet_df)
        print(adj)
        up = self.generate_user_post_matrix(tweet_df)
        print(up)
        user_pred = self.inference(user, tweet, adj, up)
        return user_pred

if __name__ == '__main__':
    sample_user_object = [
        20,
        4443,
        5021,
        2240,
        504,
        0.0,
        0.0,
        0.0,
        0.0,
        '2022/03/28 09:45:23.19485',
        '2021/01/28 13:30:34.00023',
        'Quoc Anh',
        'dirtygay2020',
        'Some description here',
    ]
    # df_naive = pd.read_csv('raw/mib/genuine_accounts.csv/tweets.csv', encoding='latin-1', escapechar='\\', header=None)
    sample_tweet_object = pd.DataFrame(
        [
            # {
            #     "id": 101,
            #     "text": "",
            #     "parent_id": 6
            # },
            # {
            #     "id": 102,
            #     "text": "",
            #     "parent_id": 6
            # },
            # {
            #     "id": i+200,
            #     "text": tweet,
            #     "parent_id": 0
            # }
            # for i, tweet in enumerate(tweets)
            # {
            #     "id": i+1,
            #     "text": str(text),
            #     "parent_id": 0
            # } for i, text in enumerate(df_naive.iloc[200:300, 1])
            # {
            #     "id": 1,
            #     "text": "Freedom endures against all odds — in the face of every aggressor — because there are always those who will fight for it. And in the 20th and 21st centuries, freedom had no greater champion than Madeleine Albright. May she always be a light to all those in the darkest places.",
            #     "parent_id": 0
            # },
            {
                "id": 2,
                "text": "@dirtygay2020 apologies if I was wrong",
                "parent_id": 0
            },
            {
                "id": 3,
                "text": "Use this http://t.co/438feiojfioj",
                "parent_id": 0
            },
            {
                "id": 4,
                "text": "This means the life-threatening experience you were in keeps happening in your nervous system. \n\nhttp://t.co/tHz1GqsjtC\n#Columbine",
                "parent_id": 3
            },
            # {
            #     "id": 5,
            #     "text": 'From the best best best @Merrillmarkoe Women on a Panel :\n\nhttps://t.co/tRVGkyPxhX\n\n#fb',
            #     "parent_id": 0,
            # },
            {
                "id": 6,
                "text": "Signing off #NiteFlirt for a while, but you can still buy my goodies at http://t.co/ay9V8DEDfE.",
                "parent_id": 2,
            },
            # {
            #     "id": 7,
            #     "text": "http://t.co/hahaha",
            #     "parent_id": 0
            # }
        ],
        columns=['id', 'text', 'parent_id']
    )
    inf = Inference()
    user_pred, tweet_pred = inf.predict(sample_user_object, sample_tweet_object)
    print('User prediction')
    print(user_pred)
    print('Tweet prediction')
    print(tweet_pred)