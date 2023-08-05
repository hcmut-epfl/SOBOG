from typing import Any, Dict
from model.gnn import GAT
import torch
import torch.nn as nn

class SOBOG(torch.nn.Module):

    def __init__(self, gpu: int, **kwargs):
        super(SOBOG, self).__init__()
        self.user_batch_norm = nn.BatchNorm1d(kwargs["n_user_features"])
        self.user_enc = nn.Linear(kwargs["n_user_features"], kwargs["d_user_embed"])
        self.post_enc = nn.Linear(kwargs["n_post_features"], kwargs["d_post_embed"])
        self.gat = nn.ModuleList([
                                    GAT(kwargs["d_post_embed"], kwargs["d_post_embed"], gpu=gpu) \
                                    for _ in range(kwargs["n_gat_layers"])
                                ])
        self.post_classifier = nn.ModuleList([
                                    nn.Linear(kwargs["d_post_embed"], kwargs["d_post_cls"]) if i==0 else
                                    nn.Linear(kwargs["d_post_cls"], 1) if i == kwargs["n_post_cls_layer"]-1 else
                                    nn.Linear(kwargs["d_post_cls"], kwargs["d_post_cls"]) for i in range(kwargs["n_post_cls_layer"])
                                ])
        self.post_aggregation = nn.AdaptiveMaxPool1d(1)
        self.user_classifier = nn.ModuleList([
                                    nn.Linear(kwargs["d_user_embed"] + kwargs["d_post_embed"], 2*kwargs["d_user_cls"]) if i==0 else
                                    nn.Linear(2*kwargs["d_user_cls"], 1) if i == kwargs["n_user_cls_layer"]-1 else
                                    nn.Linear(2*kwargs["d_user_cls"], 2*kwargs["d_user_cls"]) for i in range(kwargs["n_user_cls_layer"])
                                ])

    def forward(self, users, posts, post_adjs, up_masking):
        users = self.user_batch_norm(users)
        users_embed = self.user_enc(users)
        posts_embed = self.post_enc(posts)

        for i in range(len(self.gat)):
            posts_embed = self.gat[i](posts_embed, post_adjs)

        posts_trans = torch.transpose(posts_embed, 1, 2)
        up_posts_aggre = self.post_aggregation(posts_trans)
        up_posts_aggre = torch.flatten(up_posts_aggre, start_dim=1)
        up_embed = torch.cat([users_embed, up_posts_aggre], 1)

        for module in self.post_classifier:
            posts_embed = module(posts_embed)
        post_label = torch.sigmoid(posts_embed)        

        for module in self.user_classifier:
            up_embed = module(up_embed)
        user_label = torch.sigmoid(up_embed)

        return user_label, post_label