from model.gnn import GAT
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class SOBOG(torch.nn.Module):
    def __init__(self, gpu = 0, *args, **kwargs):
        super(SOBOG, self).__init__(*args, **kwargs)
        self.user_enc = nn.Linear(kwargs["n_user_features"], kwargs["d_user_embed"])
        self.post_enc = nn.Linear(kwargs["n_post_features"], kwargs["d_post_embed"])
        self.gat = nn.ModuleList([
                                    GAT(kwargs["d_post_embed"], kwargs["d_post_embed"], gpu=gpu) \
                                    for _ in range(kwargs["n_gat_layers"])
                                ])
        self.post_classifier = nn.ModuleList([
                                    nn.Linear(kwargs["d_post_embed"], kwargs["d_cls"]) if i==0 else
                                    nn.Linear(kwargs["d_cls"], 1) if i == kwargs["n_cls_layer"]-1 else
                                    nn.Linear(kwargs["d_cls"], kwargs["d_cls"]) for i in range(kwargs["n_cls_layer"])
                                ])
        self.post_aggregation = nn.AvgPool1d(kwargs["d_post_embed"])
        self.user_classifier = nn.ModuleList([
                                    nn.Linear(kwargs["d_user_embed"] + kwargs["d_post_embed"], 2*kwargs["d_cls"]) if i==0 else
                                    nn.Linear(2*kwargs["d_cls"], 1) if i == kwargs["n_cls_layer"]-1 else
                                    nn.Linear(2*kwargs["d_cls"], 2*kwargs["d_cls"]) for i in range(kwargs["n_cls_layer"])
                                ])

    def forward(self, users, posts, post_adjs, up_masking):
        users_embed = self.user_enc(users)
        posts_embed = self.post_enc(posts)

        for i in range(len(self.gat)):
            posts_embed = self.gat[i](posts_embed, post_adjs)

        post_label = self.post_classifier(posts_embed)

        up_posts_aggre = self.post_aggregation(posts_embed)
        up_embed = F.cat([users_embed, up_posts_aggre], 1)

        user_label = self.user_classifier(up_embed)

        return user_label, post_label

