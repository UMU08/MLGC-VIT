import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attention_weights = F.softmax(self.fc(x), dim=1)
        attended_x = torch.mul(x, attention_weights)
        return attended_x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nfeat + nhid, nhid)
        self.gc3 = GraphConvolution(nfeat + 2 * nhid, nhid)
        self.gc4 = GraphConvolution(nfeat + 3 * nhid, nhid)
        self.gc5 = GraphConvolution(nfeat + 4 * nhid, nhid)

        self.attention = AttentionLayer(5 * nhid, 5 * nhid)
        # self.gc6 = GraphConvolution(nfeat + 5 * nhid, nclass)
        self.HC = nn.Sequential(
            nn.Linear(5 * nhid, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(32, nclass)
        )

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gc2(torch.cat([x, x1], 1), adj)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(torch.cat([x, x1, x2], 1), adj)
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x4 = self.gc4(torch.cat([x, x1, x2, x3], 1), adj)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x5 = self.gc5(torch.cat([x, x1, x2, x3, x4], 1), adj)
        x1_to_x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        features = x1_to_x5.resize(2708, 5, 128)
        features = features.detach().cpu().numpy()
        attended_x = self.attention(x1_to_x5)
        attended_x = attended_x.view(attended_x.shape[0], -1)
        output = self.HC(attended_x)
        return F.log_softmax(output, dim=1),features


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nfeat+nhid, nhid)
#         self.gc3 = GraphConvolution(nfeat + 2 * nhid, nhid)
#         self.gc4 = GraphConvolution(nfeat + 3 * nhid, nhid)
#         self.gc5 = GraphConvolution(nfeat + 4 * nhid, nhid)
#         self.dropout = dropout
#         # image_size = (00, 16)
#         image_size = (20, 32)
#         patch_size = (5, 8)
#         self.vit_model = ViT(image_size=image_size,s patch_size=patch_ize, num_classes=nclass, dim=512, depth=6,
#                              heads=12,
#                              mlp_dim=512,
#                              channels=1)
#
#     def forward(self, x, adj):
#         x1 = F.relu(self.gc1(x, adj))
#         x1 = F.dropout(x1, self.dropout, training=self.training)
#         x2 = self.gc2(torch.cat([x, x1], 1), adj)
#         x2 = F.dropout(x2, self.dropout, training=self.training)
#         x3 = self.gc3(torch.cat([x, x1, x2], 1), adj)
#         x3 = F.dropout(x3, self.dropout, training=self.training)
#         x4 = self.gc4(torch.cat([x, x1, x2, x3], 1), adj)
#         x4 = F.dropout(x4, self.dropout, training=self.training)
#         x5 = self.gc5(torch.cat([x, x1, x2, x3, x4], 1), adj)
#         vit_x = torch.cat([x1, x2, x3, x4, x5], dim=1)
#         print(vit_x.shape)
#         vit_x = vit_x.reshape(vit_x.shape[0], 1, 20, 32)
#         # vit_x = vit_x.reshape(vit_x.shape[0], 1, 20, 32)
#         vit_output = self.vit_model(vit_x)
#         return F.log_softmax(vit_output, dim=1)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0.5, emb_dropout=0.5):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
