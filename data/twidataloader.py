import pickle
import re
import torch
import numpy as np
import pandas as pd
import os

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class Twibot22SampleDataset(Dataset):

    def __init__(
            self,
            sample: bool = True,
            train: bool = False,
            use_tfidf: bool = False,
            use_sbert: bool = False,
            tfidf_pretrained: bool = False,
            limit_tweets: bool = None,
            process_tweets: bool = False,
            normalize: bool = True,
            keep_indices_in_tweet_adj: bool = False,
    ):

        # Configurations
        self.sample = sample
        self.train = train
        self.use_tfidf = use_tfidf
        self.use_sbert = use_sbert
        self.normalize = normalize

        large_path = "sample" if sample else "full"
        data_path = "train" if train else "test"
        label_file = 'labels.csv'
        user_file = 'user.csv'
        edge_file = 'edge.csv'
        tweet_file = 'tweet.csv'

        print("Reading files...")
        self.user = pd.read_csv(os.path.join(large_path, data_path, user_file))
        self.label = pd.read_csv(os.path.join(large_path, data_path, label_file))
        self.edge = pd.read_csv(os.path.join(large_path, data_path, edge_file))
        self.tweet = pd.read_csv(os.path.join(large_path, data_path, tweet_file), low_memory=False)

        print("Processing user data...")
        self.ids, self.user = self.feature_user(self.user)

        if normalize:
            print("Normalizing data...")
            self.user_mean = self.user.mean(axis=0)
            self.user_std = self.user.std(axis=0)
            self.user = (self.user - self.user_mean) / self.user_std
            self.user.fillna(0, inplace=True)

        if process_tweets:
            print("Preprocessing tweets...")
            self.tweet["text"] = self.tweet["text"].apply(self.preprocessing_tweets)

        if use_tfidf:
            print("Vectorizing tweets...")
            tfidf = self.feature_tweet(self.tweet["text"], tfidf_pretrained)
            print("Processing tweets for each user...")
            self.group_user_tweets_and_matrices(self.ids, self.tweet, self.edge, keep_indices_in_tweet_adj, tfidf)
        elif use_sbert:
            print("Vectorizing tweets into SBERT...")
            sbert = self.feature_tweet_sbert(self.tweet["text"])
            with open('test.npy', 'wb') as f:
                np.save(f, sbert)
            pass
            print("Processing tweets for each user...")
            self.group_user_tweets_and_matrices(self.ids, self.tweet, self.edge, keep_indices_in_tweet_adj, sbert)
        else:
            print("Processing tweets for each user...")
            self.group_user_tweets_and_matrices(self.ids, self.tweet, self.edge, keep_indices_in_tweet_adj)

        print("Delete unused data...")
        del (self.tweet)
        del (self.edge)

    def feature_user(self, user_df: pd.DataFrame) -> pd.DataFrame:
        return user_df.pop("id"), user_df.astype('float').fillna(0.0)

    def preprocessing_tweets(self, row):
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

    def group_user_tweets_and_matrices(self, ids, tweet, edge, keep_indices, tfidf=None):
        pbar = tqdm(enumerate(ids))
        self.user_tweets = {}

        for i, user_id in pbar:

            # Filtering user's tweets
            user_tweets = tweet[tweet["author_id"] == user_id].copy(deep=True)
            user_tweets_ids = user_tweets["id"]

            user_edges = edge[edge["target_id"].isin(user_tweets_ids)].copy(deep=True)
            user_target_ids = user_edges["source_id"]

            # Filtering related tweets
            related_tweets = tweet[tweet["id"].isin(user_target_ids)].copy(deep=True)

            user_tweets["from_user"] = 1
            related_tweets["from_user"] = 0
            all_tweet_dfs = pd.concat([user_tweets, related_tweets])
            all_tweet_dfs = all_tweet_dfs[~all_tweet_dfs.index.duplicated(keep='first')]
            all_tweet_indices = all_tweet_dfs.index
            all_tweet_dfs = all_tweet_dfs.merge(self.label, how="left", left_on="author_id", right_on="id",
                                                suffixes=('', '_'))

            # Indexing tweets
            tweet_indices = {tweet_id: i for i, tweet_id in enumerate(all_tweet_dfs["id"])}
            user_edges["source_index"] = user_edges["source_id"].map(tweet_indices)
            user_edges["target_index"] = user_edges["target_id"].map(tweet_indices)

            # Create sparse matrix
            n_tweets = len(all_tweet_dfs)
            n_edges = len(user_edges)

            if keep_indices:
                tweet_adj = zip(user_edges["source_index"], user_edges["target_index"])
            else:
                tweet_adj = csr_matrix((np.ones((n_edges,)), (user_edges["source_index"], user_edges["target_index"])),
                                       shape=(n_tweets, n_tweets)).toarray()

            if tfidf is None:
                tweet_texts = all_tweet_dfs["text"].values
            else:
                tweet_texts = tfidf[all_tweet_indices]
            owned = all_tweet_dfs["from_user"].values
            tlabel = all_tweet_dfs["label"].values

            self.user_tweets[i] = (tweet_adj, tweet_texts, owned, tlabel)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index):
        user = self.user.iloc[index].values
        tweet_adj, tweet_texts, owned, tlabel = self.user_tweets[index]
        label = self.label.iloc[index]["label"]
        return user, tweet_texts, tweet_adj, owned, label, tlabel

    def feature_tweet(self, tweet_df, adapt_pretrained):
        if adapt_pretrained:
            with open('vect/vectorizer.pk', 'rb') as fin:
                vect = pickle.load(fin)
            v = vect.transform(tweet_df)
            del (vect)
        else:
            vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
            v = vect.fit_transform(tweet_df)
            with open('vect/vectorizer.pk', 'wb') as fin:
                pickle.dump(vect, fin)
            del (vect)
        return v

    def feature_tweet_sbert(self, tweet_df):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.zeros((len(tweet_df), 384))
        for i in tqdm(range(0, len(tweet_df), 32)):
            sublist = tweet_df[i:i + 32].tolist()
            embeddings[i:i + 32] = model.encode(sublist)
        return embeddings

    def save(self):
        filename = self.get_file_path(self.sample, self.train, self.use_tfidf, self.use_sbert, self.normalize)
        torch.save(self, filename)

    @staticmethod
    def get_file_path(sample, train, use_tfidf, use_sbert, normalize):
        return 'data/twibot_22_{}_{}_{}_{}.pt'.format(
            "sample" if sample else "full",
            "train" if train else "test",
            "text-vectorized" if use_tfidf else "text-sbert" if use_sbert else "text-raw",
            "normalized" if normalize else "non-normalized"
        )

