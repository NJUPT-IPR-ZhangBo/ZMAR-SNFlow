## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np
from einops import rearrange
from thop import profile


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
class Mask(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='WithBias', bias=False,):
        super(Mask, self).__init__()
        #self.temperature = nn.Parameter(torch.ones(4, 1, 1, 1))
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask_in, N):
        n11, c11, h11, w11 = x.shape
        h_pad = N - h11 % N if not h11 % N == 0 else 0
        w_pad = N - w11 % N if not w11 % N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')
        # m = F.pad(mask_in, (0, w_pad, 0, h_pad), 'reflect')
        #x = self.norm1(x)

        b, c, h, w = x.shape
        temperature = nn.Parameter(torch.ones(b, 1, 1, 1)).cuda()
       # mask_in = mask_in.expand(b, c, -1, -1)
        qkv = self.qkv_dwconv(self.qkv(x))
        _, _, v = qkv.chunk(3, dim=1)
        # mqkv = self.qkv_dwconv(self.qkv(mask_in))
        # mq, mk, mv = mqkv.chunk(3, dim=1)
        v = rearrange(v, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=N, w1=N, head=self.num_heads)
        q = rearrange(mask_in, 'b c (h1 h) (w1 w) -> b  c (h1 w1) (h w)', h1=N, w1=N) #  N^2,HW/N^2
        k = rearrange(mask_in, 'b c (h1 h) (w1 w) -> b  c (h1 w1) (h w)', h1=N, w1=N)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * temperature
        attn = attn.softmax(dim=-1)  # 通过mask得到注意力图
        attn = attn.expand(-1, c, -1, -1)
        attn = rearrange(attn, 'b (head c) h w -> b head c h w',  head=self.num_heads, c= int(c/self.num_heads))
        out = (attn @ v)
        out = rearrange(out, 'b head c (h1 w1) (h w) -> b (head c) (h1 h) (w1 w)', h=int(h/N), w=int(w/N), h1=N, w1=N, head=self.num_heads)
        out = self.project_out(out)
        out = out[:, :, :h11, :w11]
        return out



class Attentionshare(nn.Module):
    def __init__(self, dim, num_heads, bias, N=4):
        super(Attentionshare, self).__init__()
        self.N = N
        self.num_heads = num_heads
        self.temperature_5 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.temperature_4 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.share_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        x_in = x

        n11, c11, h11, w11 = x.shape
        h_pad = 4 - h11 % 4 if not h11 % 4 == 0 else 0
        w_pad = 4 - w11 % 4 if not w11 % 4 == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        v_share = rearrange(v, 'n (head c) (h1 h) (w1 w) -> n head c (h w) (h1 w1)', h1=self.N, w1=self.N, head=self.num_heads)
        q = rearrange(q, 'n (head c) h w -> n head c (h w)', head=self.num_heads)
        k = rearrange(k, 'n (head c) h w -> n head c (h w)', head=self.num_heads)
        v = rearrange(v, 'n (head c) h w -> n head c (h w)', head=self.num_heads)
        q_mask = rearrange(mask, 'n (head c) (h1 h) (w1 w) -> n head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)  # N^2,HW/N^2
        k_mask = rearrange(mask, 'n (head c) (h1 h) (w1 w) -> n head c (h1 w1) (h w)', h1=self.N, w1=self.N, head=self.num_heads)
        q_mask = torch.nn.functional.normalize(q_mask, dim=-1)
        k_mask = torch.nn.functional.normalize(k_mask, dim=-1)

        attn_mask = (q_mask @ k_mask.transpose(-2, -1)) * self.temperature_5
        attn_mask = attn_mask.softmax(dim=-1)
        attn_share = (v_share @ attn_mask)
        attn_share = rearrange(attn_share, 'n head c (h w) (h1 w1) -> n (head c) (h1 h) (w1 w)', h=int(h/self.N), w=int(w/self.N), h1=self.N, w1 = self.N, head=self.num_heads)
        attn_out = self.share_out(attn_share)  # 1x1conv
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature_4
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'n head c (h w) -> n (head c) h w', h=h, w=w, head=self.num_heads)

        out = self.project_out(out)  # 1x1conv
        out = out[:, :, :h11, :w11]
        attn_out = out[:, :, :h11, :w11]
        out = out + attn_out

        return out


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        #补上mask maskwindow
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlockWithMask(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlockWithMask, self).__init__()

        self.mask = Mask(dim, num_heads)
        self.attn_share = Attentionshare(dim, num_heads, bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask, N):
        #  ################### 串  联 #########################
        x = x + self.mask(self.norm1(x), mask, N)
        x = x + self.attn(self.norm1(x))
        out = x + self.ffn(self.norm2(x))
        return out
        # ##################################################

        # # ################### 并  联 #########################
        #
       # fea = self.mask(self.norm1(x), mask)
       # x = x + self.attn(self.norm1(x)) + fea
       # out = x + self.ffn(self.norm2(x))
      #  return out
        # # ##################################################

        ################### 共  享  v #########################

       # x = 2 * x + self.attn_share(self.norm1(x), mask)
       # out = x + self.ffn(self.norm2(x))
       # return out
        ##################################################


############################################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##############################################################################################
class TransformerBlock_1(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers=6, dim=576, num_heads=4, ffn_expansion_factor=2.66,bias=True, LayerNorm_type='WithBias'):
        # 2048
        super().__init__()

        self.layer_stack = nn.ModuleList([
            TransformerBlockWithMask(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(n_layers)])

    def forward(self, x, mask, N):
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask, N)
        return x





##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        #num_blocks = [4,4,4],
        #num_blocks=[4,4,4,4],
        num_blocks=[4, 4, 4, 4],
        num_refinement_blocks = 4,
        #num_refinement_blocks=3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # self.norm = Mask_norm(mask, dim)
        self.encoder_level1 = TransformerBlock_1(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0])
       # self.encoder_level1 = nn.Sequential(*[
      #      TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
         #                    bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1])
       # self.encoder_level2 = nn.Sequential(*[
            #TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
           #                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock_1(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[2])
      #  self.encoder_level3 = nn.Sequential(*[
         #   TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                     #        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
       # self.latent1 = TransformerBlock_1(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[3])
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = TransformerBlock_1(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[2])
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        # self.decoder_level2 = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1])
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.decoder_level1 = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0])
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        # self.refinement = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_refinement_blocks)
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.Conv2d(64, 128, 3, 2, 1), nn.Conv2d(128, 192, 3, 2, 1), nn.Sigmoid())
        self.reduce_chan0 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 2), kernel_size=1, bias=bias)

    def downmask(self, mask, k, N):
        _, _, h, w = mask.shape

        size = (int(h / k), int(w / k))
        #mask = mask.cpu()
        #mask = mask.numpy()
        #mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        #mask[mask <= 0.5] = 0
        #mask[mask > 0.5] = 1
        #mask = torch.from_numpy(mask)

        #mask = torch.unsqueeze(mask, 0)
        mask = F.interpolate(mask, size=size, mode='bilinear')
        mask[mask <= 0.2] = 0
        mask[mask > 0.2] = 1
        _, _, h, w = mask.shape
        h_pad = N - h % N if not h % N == 0 else 0
        w_pad =  N - w % N if not w % N == 0 else 0
        mask = F.pad(mask, (0, w_pad, 0, h_pad), 'reflect')
        #m_v = mask
        #m_v[mask != 0] = 255
        #m_v = torch.squeeze(m_v, 0)
        #m_v = m_v.numpy()
        #num = str(np.random.randint(1000000, 9999999))
        #out_file_name = num + ".png"
        # if k == 2:
        #     out_file_dir = "C:\\code\\Restormer-mask\\Restormer_with_MASK\\2\\" + out_file_name
        #     cv2.imwrite(out_file_dir, m_v)
        # if k == 4:
        #     out_file_dir = "C:\\code\\Restormer-mask\\Restormer_with_MASK\\3\\" + out_file_name
        #     cv2.imwrite(out_file_dir, m_v)
        # if k == 8:
        #     out_file_dir = "C:\\code\\Restormer-mask\\Restormer_with_MASK\\4\\" + out_file_name
        #     cv2.imwrite(out_file_dir, m_v)
        #mask = torch.unsqueeze(mask, 0)
        #mask = mask.expand(b, self.dim * k, -1, -1)
        #out_mask = mask.cuda()
        return mask

    def forward(self, inp_img, mask):

       # m, _, _ = torch.chunk(inp_img, 3, dim=1)
        m = mask
        b, _, _, _ = inp_img.shape
       # m_1 = m.expand(b, self.dim, -1, -1)
        # 1,48,160,160 保持mask和进来图片维度一致
        result = {}
        inp_enc_level1 = self.patch_embed(inp_img)


        #out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, m, 4)

       # out_enc_level1 = self.padd(out_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #out_enc_level2 = self.encoder_level2(inp_enc_level2)
        m_2 = self.downmask(m, int(2), 4)  # 相应下采样
        out_enc_level2 = self.encoder_level2(inp_enc_level2, m_2, 4)

       # out_enc_level2 = self.padd(out_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        #out_enc_level3 = self.encoder_level3(inp_enc_level3)
        m_3 = self.downmask(m, int(2**2),4)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, m_3, 4)

        #out_enc_level3 = self.padd(out_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        #m_4 = self.downmask(m, int(2**3),2)
        #latent = self.latent1(inp_enc_level4, m_4, 2)
        latent = self.latent(inp_enc_level4)
       # latent = self.latent2(latent)
        result['fea_up0'] = self.reduce_chan0(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3, m_3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        #result['fea_up1'] = self.reduce_chan1(out_dec_level3)
        result['fea_up1'] = out_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # out_dec_level2 = self.decoder_level2(inp_dec_level2, m_2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        result['fea_up2'] = self.reduce_chan2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        #m_11 = m.expand(b, 2 * self.dim, -1, -1)
        # out_dec_level1 = self.decoder_level1(inp_dec_level1, m_11)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)


        # out_dec_level1 = self.refinement(out_dec_level1, m_11)
        out_dec_level1 = self.refinement(out_dec_level1)
        result['cat_f'] = out_dec_level1

    #    out_dec_level0 = self.output(out_dec_level1) + inp_img
     #   result['color_map'] = self.fine_tune_color_map(out_dec_level0)
       # result['cat_f'] = out_dec_level0

        return result


# if __name__ == "__main__":
#     input_size = 256
#     num_heads = 8
#     x = torch.randn([1,3,600,400])
#     mask = torch.randn([1,3,600,400])
#     t = Restormer()
#     x = x.cuda()
#     mask = mask.cuda()
#     x = t(x,mask)

if __name__ == "__main__":
    # inputs = torch.randn([1, 3, 600, 400])
    net = Restormer()  # 定义好的网络模型
    inputs = torch.randn(1, 3, 512, 512)
    inputs1 = torch.randn(1, 1, 512, 512)
    device = torch.device('cuda:0')
    inputs = inputs.to(device)
    inputs1 = inputs1.to(device)
    flops, params = profile(net.cuda(), (inputs,inputs1,))
    print('flops: ', flops, 'params: ', params)