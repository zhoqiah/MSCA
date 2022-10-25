## from https://github.com/lucidrains/vit-pytorch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary

 # einops张量操作神器
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): ## 最重要的都是forword函数了
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        ## 对tensor张量分块 x :1 197 1024   qkv 最后 是一个元组，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # 分成多少个Head,与TRM生成qkv 的方式不同， 要更简单，不需要区分来自Encoder还是Decoder

        # 矩阵转置,k.transpose
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
# 1. VIＴ整体架构从这里开始
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # 初始化函数内，是将输入的图片，得到 img_size ，patch_size 的宽和高
        image_height, image_width = pair(image_size) ## 224*224 *3
        patch_height, patch_width = pair(patch_size)## 16 * 16  *3
        #图像尺寸必须能被patch大小整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) ## 步骤1.一个图像 分成 N 个patch
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),# 步骤2.1将patch 铺开
            nn.Linear(patch_dim, dim), # 步骤2.2 然后映射到指定的embedding的维度
        )  # 生成所有token的位置编码，包括CLS

        self.trans = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h = 7, w = 7)  # 14 14
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  ## img 1 3 224 224  输出形状x : 1 196 1024
        b, n, _ = x.shape ##
        #将cls 复制 batch_size 份
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 将cls token在维度1 扩展到输入上
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # 输入TRM
        x = self.transformer(x)

        encoder = x[:, 1:]
        encoder = self.trans(encoder)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return encoder,self.mlp_head(x)


if __name__ == '__main__':

    v = ViT(
        image_size = 224,
        patch_size = 32,
        num_classes = 2048,
        dim = 2048,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )  # patch_size = 16, heads = 16

    img = torch.randn(4, 3, 224, 224)

    encoder,preds = v(img)
    print(preds.shape)    # (4, 2048)
    print(encoder.shape)    # (4, 2048, 14, 14)

