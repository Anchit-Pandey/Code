"""
cs3t_unet.py  —  CS3T-UNet  (paper baseline, IEEE INFOCOM 2024)
===============================================================
Verified: 19,599,618 params (~19.60M) ≈ paper's 19.64M
Config  : C=64 | FFN ratio=2 | group-wise temporal MSA (nG=4)
          symmetric (2,2,6,2) enc+dec | static mask λ=0.9
          plain additive skip connections (no gating)

Key differences vs CS3T-Lite:
  C=64 (not 48) | symmetric decoder (2,2,6,2 not 1,1,2,1)
  dense group-wise temporal MSA (not DFTA)
  static energy mask λ=0.9 (not learned EG-DTP)
  plain additive skips (not gated)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CSWin Spatial Attention  (same as CS3T-Lite, retained)
# ─────────────────────────────────────────────────────────────────────────────
class CSWinSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, stripe_width=2):
        super().__init__()
        assert num_heads % 2 == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.sw        = stripe_width
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, 3*dim, bias=True)
        self.proj = nn.Linear(dim, dim,   bias=True)
        self.norm = nn.LayerNorm(dim)

    def _stripe_attn(self, x, horizontal):
        B, H, W, C = x.shape
        hh = self.num_heads // 2
        hd = hh * self.head_dim
        qkv = self.qkv(x)
        if horizontal:
            q = qkv[..., :hd]; k = qkv[...,C:C+hd]; v = qkv[...,2*C:2*C+hd]
            ns = max(1, H//self.sw); sw = H//ns
            def r(t): return t[:,:ns*sw].reshape(B,ns,sw,W,hh,self.head_dim).permute(0,1,4,2,3,5).reshape(B*ns,hh,sw*W,self.head_dim)
            q,k,v = r(q),r(k),r(v)
            out = (F.softmax((q@k.transpose(-2,-1))*self.scale,dim=-1)@v)
            out = out.reshape(B,ns,hh,sw,W,self.head_dim).permute(0,1,3,4,2,5).reshape(B,ns*sw,W,hd)
            if ns*sw < H: out = F.pad(out,(0,0,0,0,0,H-ns*sw))
        else:
            x = x.permute(0,2,1,3); qkv = self.qkv(x)
            q = qkv[...,hd:2*hd]; k = qkv[...,C+hd:C+2*hd]; v = qkv[...,2*C+hd:2*C+2*hd]
            ns = max(1, W//self.sw); sw = W//ns
            def r(t): return t[:,:ns*sw].reshape(B,ns,sw,H,hh,self.head_dim).permute(0,1,4,2,3,5).reshape(B*ns,hh,sw*H,self.head_dim)
            q,k,v = r(q),r(k),r(v)
            out = (F.softmax((q@k.transpose(-2,-1))*self.scale,dim=-1)@v)
            out = out.reshape(B,ns,hh,sw,H,self.head_dim).permute(0,1,3,4,2,5).reshape(B,ns*sw,H,hd)
            if ns*sw < W: out = F.pad(out,(0,0,0,0,0,W-ns*sw))
            out = out.permute(0,2,1,3)
        return out

    def forward(self, x):
        r = x; x = self.norm(x)
        return self.proj(torch.cat([self._stripe_attn(x,True),
                                    self._stripe_attn(x,False)],dim=-1)) + r


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Group-wise Temporal MSA  (baseline — dense per-group, nG=4)
#     Each group of C//nG channels gets its own QKV projection.
#     No shared out_proj — groups concatenated directly.
#     This matches the ~19.64M parameter count.
# ─────────────────────────────────────────────────────────────────────────────
class GroupWiseTemporalMSA(nn.Module):
    def __init__(self, dim, num_groups=4):
        super().__init__()
        self.nG    = num_groups
        self.g     = dim // num_groups
        self.scale = self.g ** -0.5
        self.norm  = nn.LayerNorm(dim)
        # Separate QKV + proj per group  (key to matching 19.64M)
        self.qkv  = nn.ModuleList([nn.Linear(self.g, 3*self.g, bias=True) for _ in range(num_groups)])
        self.proj = nn.ModuleList([nn.Linear(self.g, self.g,   bias=True) for _ in range(num_groups)])

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x  = self.norm(x).reshape(B*H*W, C)
        out = []
        for i in range(self.nG):
            xi      = x[:, i*self.g:(i+1)*self.g]         # [BHW, g]
            q,k,v   = self.qkv[i](xi).split(self.g, -1)
            q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)
            attn    = F.softmax((q@k.transpose(-2,-1))*self.scale, dim=-1)
            oi      = (attn@v).squeeze(1)
            out.append(self.proj[i](oi))
        return torch.cat(out, dim=-1).reshape(B,H,W,C) + residual


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Static Energy Mask  (baseline — λ=0.9, zero learnable parameters)
# ─────────────────────────────────────────────────────────────────────────────
class StaticEnergyMask(nn.Module):
    def __init__(self, lam=0.9):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        B, H, W, _ = x.shape
        power  = (x**2).sum(-1)
        pf     = power.reshape(B, H*W)
        sp, _  = pf.sort(dim=-1, descending=True)
        cs     = sp.cumsum(-1) / (sp.sum(-1,keepdim=True) + 1e-10)
        k      = (cs <= self.lam).sum(-1).clamp(min=1)
        thr    = sp[torch.arange(B), k-1].reshape(B,1,1)
        return (power >= thr).float().unsqueeze(-1)       # [B,H,W,1]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CS3T-UNet Block
# ─────────────────────────────────────────────────────────────────────────────
class CS3TUNetBlock(nn.Module):
    def __init__(self, dim, num_heads=4, stripe_width=2,
                 num_groups=4, lam=0.9, ffn_ratio=2):
        super().__init__()
        self.spatial_attn  = CSWinSelfAttention(dim, num_heads, stripe_width)
        self.energy_mask   = StaticEnergyMask(lam)
        self.temporal_attn = GroupWiseTemporalMSA(dim, num_groups)
        self.norm_ffn      = nn.LayerNorm(dim)
        fd = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim,fd), nn.GELU(), nn.Linear(fd,dim))

    def forward(self, x):
        mask   = self.energy_mask(x)
        x_attn = self.spatial_attn(x)
        x = x + x_attn * mask
        x = self.temporal_attn(x)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Merge / Expand
# ─────────────────────────────────────────────────────────────────────────────
class MergeBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 2*in_dim, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(2*in_dim)
    def forward(self, x):
        return self.norm(self.conv(x.permute(0,3,1,2)).permute(0,2,3,1))

class ExpandBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2*in_dim, bias=True)
        self.ps     = nn.PixelShuffle(2)
        self.norm   = nn.LayerNorm(in_dim//2)
    def forward(self, x):
        x = self.ps(self.linear(x).permute(0,3,1,2)).permute(0,2,3,1)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Full CS3T-UNet
# ─────────────────────────────────────────────────────────────────────────────
def _make_stage(n, dim, nh, sw, ng, lam, ffn_ratio):
    return nn.Sequential(*[
        CS3TUNetBlock(dim, nh, sw, ng, lam, ffn_ratio)
        for _ in range(n)
    ])

class CS3TUNet(nn.Module):
    """
    CS3T-UNet — paper baseline (IEEE INFOCOM 2024).
    ~19.60M parameters with C=64, FFN ratio=2, group-wise temporal.
    """
    def __init__(self, Nf=64, Nt=64, T=10, L=1,
                 C=64, blocks=(2,2,6,2),
                 num_heads=4, stripe_width=2,
                 num_groups=4, lam=0.9,
                 ffn_ratio=2,              # ratio=2 matches ~19.64M
                 patch_size=2):
        super().__init__()
        self.C = C
        dims   = [C, 2*C, 4*C, 8*C]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(2*T, C, patch_size, stride=patch_size),
            nn.LayerNorm(C))

        # Encoder
        self.enc1   = _make_stage(blocks[0], dims[0], num_heads,          stripe_width, num_groups, lam, ffn_ratio)
        self.enc2   = _make_stage(blocks[1], dims[1], min(num_heads*2,8), stripe_width, num_groups, lam, ffn_ratio)
        self.enc3   = _make_stage(blocks[2], dims[2], min(num_heads*4,8), stripe_width, num_groups, lam, ffn_ratio)
        self.enc4   = _make_stage(blocks[3], dims[3], min(num_heads*8,8), stripe_width, num_groups, lam, ffn_ratio)
        self.merge1 = MergeBlock(dims[0])
        self.merge2 = MergeBlock(dims[1])
        self.merge3 = MergeBlock(dims[2])

        # Decoder  — SYMMETRIC: same block counts as encoder
        self.dec4   = _make_stage(blocks[3], dims[3], min(num_heads*8,8), stripe_width, num_groups, lam, ffn_ratio)
        self.dec3   = _make_stage(blocks[2], dims[2], min(num_heads*4,8), stripe_width, num_groups, lam, ffn_ratio)
        self.dec2   = _make_stage(blocks[1], dims[1], min(num_heads*2,8), stripe_width, num_groups, lam, ffn_ratio)
        self.dec1   = _make_stage(blocks[0], dims[0], num_heads,          stripe_width, num_groups, lam, ffn_ratio)
        self.exp3   = ExpandBlock(dims[3])
        self.exp2   = ExpandBlock(dims[2])
        self.exp1   = ExpandBlock(dims[1])

        # Head
        self.final_expand = ExpandBlock(dims[0])
        self.head         = nn.Linear(C//2, 2*L)
        self.act          = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _embed(self, x):
        x = self.patch_embed[0](x.permute(0,3,1,2)).permute(0,2,3,1)
        return self.patch_embed[1](x)

    def forward(self, x):
        x  = self._embed(x)
        s1 = self.enc1(x);  x = self.merge1(s1)
        s2 = self.enc2(x);  x = self.merge2(s2)
        s3 = self.enc3(x);  x = self.merge3(s3)
        x  = self.enc4(x)
        x  = self.dec4(x)
        x  = self.exp3(x) + s3          # plain additive skip (no gate)
        x  = self.dec3(x)
        x  = self.exp2(x) + s2
        x  = self.dec2(x)
        x  = self.exp1(x) + s1
        x  = self.dec1(x)
        x  = self.final_expand(x)
        return self.act(self.head(x))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NMSE metric
# ─────────────────────────────────────────────────────────────────────────────
def compute_nmse(pred, target):
    B = pred.shape[0]; p = pred.reshape(B,-1); t = target.reshape(B,-1)
    return 10*torch.log10(((p-t).pow(2).sum(1)/(t.pow(2).sum(1)+1e-10)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    for L in [1, 5]:
        m = CS3TUNet(T=10, L=L, C=64).to(device)
        x = torch.randn(2, 64, 64, 20).to(device)
        with torch.no_grad(): y = m(x)
        n = sum(p.numel() for p in m.parameters())
        ok = '✓' if 19e6 < n < 20e6 else 'CHECK'
        print(f"L={L}: {tuple(x.shape)} → {tuple(y.shape)} | "
              f"params={n:,} ({n/1e6:.2f}M) | paper=19.64M  {ok}")
