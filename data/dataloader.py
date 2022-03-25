import re
import scipy
import torch
import datetime
import pickle
from time import sleep
from more_itertools import sample
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict
from configparser import RawConfigParser
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.feature_extraction.text import TfidfVectorizer

class TwitterDataset(Dataset):

    def __init__(self, tfidf_pretrained=False, limit_tweets=None, normalize=True):
        print("Getting configuration...")
        config = self.get_data_config()

        print("Loading user data...")
        self.user, self.ids, self.label = self.get_user_data(config)

        print("Featuring user data...")
        self.user = self.feature_user(self.user)

        if normalize:
            print("Normalizing data...")
            self.user_mean = self.user.mean(axis=0)
            self.user_std = self.user.std(axis=0)
            self.user = (self.user - self.user_mean) / self.user_std

        print("Loading tweet data...")
        self.tweet, tweet_metadata, self.tweet_label = self.get_tweet_data(config, limit_tweets)

        print("Preprocessing tweet...")
        self.tweet = self.tweet.apply(self.preprocessing)

        print("Vectorizing tweets...")
        self.tweet = self.feature_tweet(self.tweet, tfidf_pretrained)

        # print("Removing user not having tweet...")
        # self.user = self.remove_user(tweet_metadata, self.user)

        print("Generating tweeting graph... Skipping...")
        # self.tweet_adj, self.utr_matrix, self.up_matrix = self.generate_adjacency_matrix(tweet_metadata, self.ids)
        # self.tweet_adj, self.utr_matrix, self.up_matrix = self.generate_adjacency_matrix(self.tweet, self.user)

        print("Converting user dataframe to numpy...")
        self.user = self.user.to_numpy()

    def convert_long_date(self, str):
        """Convert Long Java string to Datetime format."""
        try:
            f = float(str[:-1]) / 1000.0
            dt_format = datetime.datetime.fromtimestamp(f)
            return dt_format
        except:
            return str

    def get_data_config(self) -> Dict:
        data_parser = RawConfigParser()
        data_config_file = './config/data.cfg'
        data_parser.read(data_config_file)
        return dict(data_parser.items('MIB'))

    def get_user_data(self, config: Dict) -> pd.DataFrame:
        """Acquire user dataframe"""
        paths_bot_user = [
                'social_spambots_1_users',
                'social_spambots_2_users',
                'social_spambots_3_users',
                'traditional_spambots_1_users',
        ]
        path_human = 'genuine_users'
        dtypes_format = {
            'updated': 'datetime64[ns]',
            'created_at': 'datetime64[ns]',
            'timestamp': 'datetime64[ns]',
            'crawled_at': 'datetime64[ns]'
        }

        df_bot_users = pd.concat(
            [pd.read_csv(config[path]) for path in paths_bot_user]
        ).reset_index(drop=True)
        df_bot_users['created_at'] = df_bot_users['created_at'].apply(self.convert_long_date)
        df_bot_users = df_bot_users.astype(dtype=dtypes_format)
        df_naive_users = pd.read_csv(config[path_human])
        label = np.concatenate([
                    np.zeros((df_bot_users.shape[0],)),
                    np.ones((df_naive_users.shape[0],))
                ])
        df_users = pd.concat([df_bot_users, df_naive_users], ignore_index=True)
        df_ids = df_users.pop('id')
        return df_users, df_ids, label

    def feature_user(self, user_df) -> pd.DataFrame:
        kept_features = [
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
        user_df = user_df[kept_features].copy()
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
        return user_df.select_dtypes('number').fillna(0.0)

    def get_tweet_data(self, config, limit_tweets) -> pd.DataFrame:
        paths_bot = [
                'social_spambots_1_tweets',
                'social_spambots_2_tweets',
                'social_spambots_3_tweets',
                'traditional_spambots_1_tweets'
        ]
        path_human = 'genuine_tweets'
        replace_map_dict = {
            "True": 1,
            "true": 1,
            "False": 0,
            "false": 0,
            "N": np.nan,
        }
        df_bot_tweets = pd.concat([
            pd.read_csv(
                config[path],
                nrows=limit_tweets,
                encoding='latin-1'
            ).replace(replace_map_dict) for path in paths_bot
        ]).reset_index(drop=True)
        df_naive_tweets = pd.read_csv(
            config[path_human],
            header=None,
            escapechar='\\',
            nrows=limit_tweets,
            encoding='latin-1'
        )
        df_naive_tweets.drop(12, axis=1, inplace=True)
        df_naive_tweets.columns = df_bot_tweets.columns
        df_tweets = pd.concat([df_bot_tweets, df_naive_tweets], ignore_index=True)
        df_tweets['text'] = df_tweets['text'].fillna('')
        label = np.concatenate([
                    np.zeros((df_bot_tweets.shape[0],)),
                    np.ones((df_naive_tweets.shape[0],))
                ])
        return df_tweets['text'], df_tweets.drop('text', axis=1), label

    def preprocessing(self, row):
        URL_PATTERN = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        rowlist = str(row).split()
        rowlist = [word.strip() for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '#') else "hashtagtag" for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '@') else "usertag" for word in rowlist]
        rowlist = [word.lower() for word in rowlist]
        rowlist = [re.sub(URL_PATTERN, "urltag", word) for word in rowlist]
        return " ".join(rowlist)

    def feature_tweet(self, tweet_df, adapt_pretrained):
        if adapt_pretrained:
            with open('vectorizer.pk', 'rb') as fin:
                vect = pickle.load(fin)
            v = vect.transform(tweet_df)
        else:
            vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
            v = vect.fit_transform(tweet_df)
            with open('vectorizer.pk', 'wb') as fin:
                pickle.dump(vect, fin)
        return v

    def remove_user(self, tweet_metadata, user):
        keep_user = tweet_metadata['user_id'].unique()
        return user[user['id'].isin(keep_user)]

    def generate_adjacency_matrix(self, tweet_metadata, ids):
        n_user = len(ids)
        edges = pd.DataFrame()

        edges['tweet_id'] = tweet_metadata['id']
        edges['user_id'] = tweet_metadata['user_id']
        
        # Combine reply and retweet id into a column (since a tweet can only be reply or retweet
        # and if not, the column will be 0)
        tweet_reply_id = tweet_metadata['in_reply_to_status_id']
        tweet_retweet_id = tweet_metadata['retweeted_status_id']
        edges['linked_id'] = tweet_reply_id + tweet_retweet_id

        rel = pd.merge(edges, edges, how='inner', left_on='linked_id', right_on='tweet_id')
        rel['is_x_exists'] = rel['user_id_x'].isin(ids)
        rel['is_y_exists'] = rel['user_id_y'].isin(ids)
        rel[['user_id_x', 'is_x_exists', 'user_id_y', 'is_y_exists']].to_csv('adp.csv', index=False)

        graph = nx.DiGraph()
        graph.add_edges_from(edges[['linked_id', 'tweet_id']].values)
        graph.remove_node(0)

        tweet_adj_matrix = nx.adjacency_matrix(
            graph,
            nodelist=edges['tweet_id'].values
        )

        ut_graph = nx.DiGraph()
        up_graph = nx.DiGraph()
        ut_graph.add_nodes_from(ids.values)
        up_graph.add_nodes_from(ids.values)
        for user_id, tweet_id in edges[['user_id', 'tweet_id']].values:
            related_tweet_ids = set(nx.nodes(nx.dfs_tree(graph, tweet_id)))
            up_graph.add_edge(user_id, tweet_id)
            for id in related_tweet_ids:
                ut_graph.add_edge(user_id, id)
        # In short: user-tweet relation matrix
        utr_matrix = nx.adjacency_matrix(
            ut_graph,
            nodelist=np.append(
                ids.values,
                edges['tweet_id'].values
            )
        )
        utr_matrix = utr_matrix[:n_user, n_user:]

        # In short: user-post matrix
        up_matrix = nx.adjacency_matrix(
            up_graph,
            nodelist=np.append(
                ids.values,
                edges['tweet_id'].values
            )
        )
        up_matrix = up_matrix[:n_user, n_user:]

        return tweet_adj_matrix, utr_matrix, up_matrix

    def convert_bsr_to_sparse_torch(self, bsr: scipy.sparse.coo_matrix):
        bsr = bsr.tocoo()
        return torch.sparse_coo_tensor(
            indices = [bsr.row, bsr.col],
            values = bsr.data
        )

    def __getitem__(self, idx):
        user = self.user[idx]
        tweet_rel = self.utr_matrix[idx].A[0]
        selected_tweet = tweet_rel == 1
        tweet = self.tweet[selected_tweet].A
        tlabel = self.tweet_label[selected_tweet]
        adj = self.tweet_adj[np.ix_(selected_tweet, selected_tweet)].A
        up = self.up_matrix[idx, selected_tweet].A
        label = self.label[idx]
        return user, tweet, adj, up, label, tlabel
    
    def __len__(self):
        return len(self.user)

if __name__ == "__main__":
    # dataset = TwitterDataset(tfidf_pretrained=False, limit_tweets=None)
    # torch.save(dataset, 'data/dataset_tfidf.pt')
    dataset = torch.load('data/dataset_full.pt')
    tf = torch.load('data/dataset_tfidf.pt')
    dataset.tweet = tf.tweet
    idx = 0
    sample_user = dataset[idx]
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
    torch.save(dataset, 'data/dataset_full.pt')
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)