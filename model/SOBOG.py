from typing import Any, Dict
from model.gnn import GAT
from model.account_graph import RGTLayer
import torch
import torch.nn as nn

from model.multihop import MultiHopGATConv


class SOBOG(torch.nn.Module):

    def __init__(self, gpu: int, **kwargs):
        super(SOBOG, self).__init__()
        self.user_enc = nn.Linear(kwargs["n_user_features"], kwargs["d_user_embed"])
        self.post_enc = nn.Linear(kwargs["n_post_features"], kwargs["d_post_embed"])
        self.gat = nn.ModuleList([
            MultiHopGATConv(kwargs["d_post_embed"], kwargs["d_post_embed"], n_hop=5, heads=2, concat=False,
                            add_self_loops=False)
            for _ in range(kwargs["n_gat_layers"])
        ])
        self.post_classifier = nn.ModuleList([
            nn.Linear(kwargs["d_post_embed"], kwargs["d_post_cls"]) if i == 0 else
            nn.Linear(kwargs["d_post_cls"], 1) if i == kwargs["n_post_cls_layer"] - 1 else
            nn.Linear(kwargs["d_post_cls"], kwargs["d_post_cls"]) for i in range(kwargs["n_post_cls_layer"])
        ])
        self.post_aggregation = nn.AdaptiveMaxPool1d(1)
        self.user_classifier = nn.ModuleList([
            nn.Linear(kwargs["d_user_embed"] + kwargs["d_post_embed"], 2 * kwargs["d_user_cls"]) if i == 0 else
            nn.Linear(2 * kwargs["d_user_cls"], 1) if i == kwargs["n_user_cls_layer"] - 1 else
            nn.Linear(2 * kwargs["d_user_cls"], 2 * kwargs["d_user_cls"]) for i in range(kwargs["n_user_cls_layer"])
        ])

    def get_index(self, post_adjs):
        out = []
        for i in range(len(post_adjs)):
            for j in range(len(post_adjs[i])):
                if post_adjs[i][j] == 0:
                    out += [[i, j]]
        return torch.tensor(out)

    def forward(self, users, posts, post_adjs, up_masking):
        users_embed = self.user_enc(users)
        posts_embed = self.post_enc(posts)
        adjs = self.get_index(post_adjs.type(torch.int)[0])
        for i in range(len(self.gat)):
            posts_embed = self.gat[i](posts_embed[0], torch.transpose(adjs, 0, 1))

        posts_embed = torch.reshape(posts_embed, (1, posts_embed.shape[0], posts_embed.shape[1]))
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