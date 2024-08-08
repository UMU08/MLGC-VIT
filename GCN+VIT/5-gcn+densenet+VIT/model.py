import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import config
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

drop_rate = config.drop_rate
batch_size = config.batch_size

# used for supervised
class JointlyTrainModel(nn.Module):
    def __init__(self, inchannel, outchannel, batch, testmode=False, **kwargs):
        super(JointlyTrainModel, self).__init__()
        self.batch = batch
        self.testmode = testmode
        linearsize = 512

        outchannel = 64

        # K = [1,2,3,4,5,6,7,8,9,10]
        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=config.K)
        self.conv2 = gnn.ChebConv(inchannel+outchannel, outchannel, K=config.K)
        self.conv3 = gnn.ChebConv(inchannel+2*outchannel, outchannel, K=config.K)
        self.conv4 = gnn.ChebConv(inchannel+3*outchannel, outchannel, K=config.K)
        self.conv5 = gnn.ChebConv(inchannel+4*outchannel, outchannel, K=config.K)

        self.bn = nn.BatchNorm1d(outchannel)
        self.relu = nn.ReLU(inplace=True)

        self.HC = nn.Sequential(
            nn.Linear(outchannel * 62, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize // 2, kwargs['HC'])
        )
        image_size = (310, 64)
        patch_size = (10, 8)
        self.vit_model = ViT(image_size = image_size, patch_size = patch_size, num_classes=3, dim=512, depth=12, heads=8, mlp_dim=1024,
                             channels=1)

    def forward(self, *args):
        if not self.testmode:
            x, e = args[0].x, args[0].edge_index  # original graph data

            x1 = self.relu(self.conv1(x, e))
            x2 = self.relu(self.conv2(torch.cat([x, x1], 1), e))
            x3 = self.relu(self.conv3(torch.cat([x, x1, x2], 1), e))
            x4 = self.relu(self.conv4(torch.cat([x, x1, x2, x3], 1), e))
            x5 = self.relu(self.conv5(torch.cat([x, x1, x2, x3, x4], 1), e))

            x1 = x1.reshape(batch_size, 62, 64)
            x2 = x2.reshape(batch_size, 62, 64)
            x3 = x3.reshape(batch_size, 62, 64)
            x4 = x4.reshape(batch_size, 62, 64)
            x5 = x5.reshape(batch_size, 62, 64)
            vit_x = torch.cat([x1, x2, x3, x4, x5], dim=1)
            vit_x = vit_x.reshape(batch_size, 1, 310, 64)
            vit_output = self.vit_model(vit_x)
            return vit_output

        else:
            xx, e3 = args[0].x, args[0].edge_index  # original graph data
            x1 = self.relu(self.conv1(xx, e3))
            x2 = self.relu(self.conv2(torch.cat([xx, x1], 1), e3))
            x3 = self.relu(self.conv3(torch.cat([xx, x1, x2], 1), e3))
            x4 = self.relu(self.conv4(torch.cat([xx, x1, x2, x3], 1), e3))
            x5 = self.relu(self.conv5(torch.cat([xx, x1, x2, x3, x4], 1), e3))

            x1 = x1.reshape(batch_size, 62, 64)
            x2 = x2.reshape(batch_size, 62, 64)
            x3 = x3.reshape(batch_size, 62, 64)
            x4 = x4.reshape(batch_size, 62, 64)
            x5 = x5.reshape(batch_size, 62, 64)
            vit_x = torch.cat([x1, x2, x3, x4, x5], dim=1)
            vit_x = vit_x.reshape(batch_size, 1, 310, 64)
            vit_output = self.vit_model(vit_x)
            return vit_output


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
