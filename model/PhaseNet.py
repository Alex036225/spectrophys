import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        y = F.pad(x, (pad, 0))
        y = self.conv(y)
        y = self.act(y)
        y = self.drop(y)
        return y + x


class TemporalCausalConvMinimal(nn.Module):
    def __init__(self, input_size, output_size, hidden=128,
                 num_layers=3, kernel_size=3, dropout=0.1, dilation_base=2):
        super().__init__()
        self.inp = nn.Conv1d(input_size, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden, kernel_size=kernel_size, dilation=(dilation_base ** i), dropout=dropout)
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(hidden, output_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x.permute(0, 2, 1)


class TIM(nn.Module):
    def __init__(self, channels, future_ratio=1 / 8, past_ratio=1 / 8):
        super().__init__()
        future_channels = int(channels * future_ratio)
        past_channels = int(channels * past_ratio)
        self.future_channels = max(future_channels, 0)
        self.past_channels = max(past_channels, 0)

    def forward(self, x):
        batch_size, channels, frames, height, width = x.shape
        future_channels = self.future_channels
        past_channels = self.past_channels
        if future_channels + past_channels == 0 or frames == 1:
            return x

        output = torch.zeros_like(x)
        if channels > future_channels + past_channels:
            output[:, future_channels + past_channels:, ...] = x[:, future_channels + past_channels:, ...]
        if future_channels > 0:
            output[:, :future_channels, :-1, ...] = x[:, :future_channels, 1:, ...]
        if past_channels > 0:
            output[:, future_channels:future_channels + past_channels, 1:, ...] = (
                x[:, future_channels:future_channels + past_channels, :-1, ...]
            )
        return output


class MixStyle3D(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not self.training or self.p <= 0.0 or x.shape[0] < 2:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sigma = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x - mu) / (sigma + self.eps)
        perm = torch.randperm(x.shape[0], device=x.device)
        mu2 = mu[perm]
        sigma2 = sigma[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1, 1, 1, 1)).to(x.device, x.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu2
        sigma_mix = lam * sigma + (1.0 - lam) * sigma2
        return x_norm * sigma_mix + mu_mix


class DatasetFiLM(nn.Module):
    def __init__(self, num_domains, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(int(num_domains), hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, domain_ids):
        if domain_ids is None:
            return x
        cond = self.mlp(self.embedding(domain_ids))
        gamma, beta = cond.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x + torch.tanh(self.gate) * (x * gamma + beta)


class TaskSeparatedDatasetFiLM(nn.Module):
    def __init__(self, num_domains, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.shared = DatasetFiLM(num_domains, feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.hr = DatasetFiLM(num_domains, feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.rr = DatasetFiLM(num_domains, feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, z_seq, hr_seq, rr_seq, domain_ids):
        return (
            self.shared(z_seq, domain_ids),
            self.hr(hr_seq, domain_ids),
            self.rr(rr_seq, domain_ids),
        )


class SupportConditionedTaskFiLM(nn.Module):
    def __init__(self, context_dim, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.context_dim = int(context_dim)
        self.feature_dim = int(feature_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.feature_dim * 6),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _match_batch(self, support_context, batch_size, device, dtype):
        if support_context is None:
            return None
        context = support_context.to(device=device, dtype=dtype)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1)
        if context.shape[0] != batch_size:
            raise ValueError(
                f"support_context batch {context.shape[0]} does not match input batch {batch_size}"
            )
        return context

    def _apply_film(self, x, gamma, beta):
        return x + torch.tanh(self.gate) * (x * torch.tanh(gamma).unsqueeze(1) + beta.unsqueeze(1))

    def forward(self, z_seq, hr_seq, rr_seq, support_context):
        context = self._match_batch(support_context, z_seq.shape[0], z_seq.device, z_seq.dtype)
        if context is None:
            return z_seq, hr_seq, rr_seq
        z_gamma, z_beta, hr_gamma, hr_beta, rr_gamma, rr_beta = self.mlp(context).chunk(6, dim=-1)
        return (
            self._apply_film(z_seq, z_gamma, z_beta),
            self._apply_film(hr_seq, hr_gamma, hr_beta),
            self._apply_film(rr_seq, rr_gamma, rr_beta),
        )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.avg_pool(x)
        weights = self.fc(weights)
        return x * weights.expand_as(x)


class EfficientSpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel=3, expand_ratio=4,
                 tsm_forward_ratio=1 / 8, tsm_backward_ratio=1 / 8):
        super().__init__()
        self.use_residual = in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.tsm = TIM(hidden_dim, future_ratio=tsm_forward_ratio, past_ratio=tsm_backward_ratio)
        self.stage2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.se = SELayer(hidden_dim)
        self.proj = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels)
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        shortcut = x
        x = self.stage1(x)
        x = self.tsm(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.se(x)
        main_out = self.proj(x)
        if self.use_residual:
            return self.pool(main_out + shortcut)
        return self.pool(main_out)


class TemporalPyramidRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False)
        self.dw5 = nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=channels, bias=False)
        self.dw7 = nn.Conv3d(channels, channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), groups=channels, bias=False)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.GELU()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.dw3(x) + self.dw5(x) + self.dw7(x)
        y = self.act(self.norm(y))
        y = self.proj(y)
        return y


class SpatialAttentionHead(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        batch_size, channels, frames, height, width = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        raw_attention = self.attention_conv(x_reshaped)
        raw_attention = raw_attention.reshape(batch_size * frames, 1, height * width)
        attention_weights = F.softmax(raw_attention, dim=2)
        attention_map = attention_weights.reshape(batch_size * frames, 1, height, width)
        attention_map = attention_map.reshape(batch_size, frames, 1, height, width).permute(0, 2, 1, 3, 4)
        return attention_map


class Decoder1D(nn.Module):
    def __init__(self, latent_dim=32, feature_dim=128, start_len=8, num_blocks=3):
        super().__init__()
        self.start_channels = feature_dim // (2 ** (num_blocks - 1))
        if self.start_channels == 0:
            self.start_channels = 1
        self.start_len = start_len
        self.initial_dense = nn.Sequential(
            nn.Linear(latent_dim, self.start_channels * start_len),
            nn.ReLU(inplace=True)
        )
        blocks = []
        in_channels = self.start_channels
        for _ in range(num_blocks):
            out_channels = in_channels * 2
            blocks.append(nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1))
            blocks.append(nn.InstanceNorm1d(out_channels))
            blocks.append(nn.ReLU())
            in_channels = out_channels
        blocks.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1))
        self.deconv = nn.Sequential(*blocks)
        final_len = start_len * (2 ** num_blocks)
        self.final_fc = nn.Linear(final_len * in_channels, feature_dim)

    def forward(self, latent):
        x = self.initial_dense(latent).reshape(-1, self.start_channels, self.start_len)
        x = self.deconv(x).flatten(1)
        return self.final_fc(x)


class SequenceRegressor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z_seq):
        out, _ = self.gru(z_seq)
        return self.head(out).squeeze(-1)


class LongRangeTemporalMixer(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.in_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (5, 2), (7, 4), (7, 8), (9, 16), (9, 32)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.out_proj = nn.Conv1d(hidden_dim, feature_dim, kernel_size=1)

    def forward(self, z_seq):
        x = self.in_proj(z_seq.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        x_seq = x.permute(0, 2, 1)
        attn_in = self.attn_norm(x_seq)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x_seq = x_seq + attn_out
        return z_seq + self.out_proj(x_seq.permute(0, 2, 1)).permute(0, 2, 1)


class BottleneckTemporalRepresentationRefiner(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=32, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.down = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        groups = 8 if bottleneck_dim % 8 == 0 else 1
        self.branches = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (9, 1), (9, 2), (15, 2), (17, 4)]:
            padding = (kernel_size // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        bottleneck_dim,
                        bottleneck_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=bottleneck_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, bottleneck_dim),
                    nn.GELU(),
                )
            )
        self.mix = nn.Sequential(
            nn.Conv1d(bottleneck_dim * len(self.branches), hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(bottleneck_dim)
        num_heads = 4 if bottleneck_dim % 4 == 0 else 1
        self.attn = nn.MultiheadAttention(bottleneck_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(bottleneck_dim)
        self.up = nn.Sequential(
            nn.Linear(bottleneck_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_seq):
        z_small = self.down(self.pre_norm(z_seq))
        x = z_small.permute(0, 2, 1)
        multi_scale = self.mix(torch.cat([branch(x) for branch in self.branches], dim=1)).permute(0, 2, 1)
        z_small = z_small + multi_scale
        attn_in = self.attn_norm(z_small)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        z_small = z_small + attn_out
        return z_seq + torch.tanh(self.residual_gate) * self.up(self.out_norm(z_small))


class VitalBandRepresentationAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1, bias=False)
        self.task_embeddings = nn.ParameterDict(
            {
                "shared": nn.Parameter(torch.zeros(1, hidden_dim, 1)),
                "hr": nn.Parameter(torch.zeros(1, hidden_dim, 1)),
                "rr": nn.Parameter(torch.zeros(1, hidden_dim, 1)),
            }
        )
        self.task_scale = nn.Parameter(torch.tensor(0.0))
        self.hr_branch = self._make_branch(hidden_dim, groups, [(5, 1), (7, 2), (9, 4), (11, 8)], dropout)
        self.rr_branch = self._make_branch(hidden_dim, groups, [(15, 1), (21, 2), (31, 4), (41, 6)], dropout)
        self.mix = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=1, bias=False),
        )
        self.out_norm = nn.LayerNorm(feature_dim)
        self.hr_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _make_branch(hidden_dim, groups, specs, dropout):
        blocks = []
        for kernel_size, dilation in specs:
            padding = (kernel_size // 2) * dilation
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_branch(x, blocks):
        y = x
        for block in blocks:
            y = y + block(y)
        return y

    def forward(self, z_seq, task="shared"):
        x = self.input_proj(self.input_norm(z_seq).permute(0, 2, 1))
        task_key = task if task in self.task_embeddings else "shared"
        x = x + torch.tanh(self.task_scale) * self.task_embeddings[task_key]
        hr = self._run_branch(x, self.hr_branch) * torch.sigmoid(self.hr_gate)
        rr = self._run_branch(x, self.rr_branch) * torch.sigmoid(self.rr_gate)
        delta = self.mix(torch.cat([hr, rr], dim=1)).permute(0, 2, 1)
        return z_seq + torch.tanh(self.residual_gate) * self.out_norm(delta)


class CrossTaskPhysioAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        heads = 4 if feature_dim % 4 == 0 else 1
        self.shared_norm = nn.LayerNorm(feature_dim)
        self.hr_norm = nn.LayerNorm(feature_dim)
        self.rr_norm = nn.LayerNorm(feature_dim)
        self.shared_tokens = nn.Parameter(torch.zeros(1, 4, feature_dim))
        nn.init.trunc_normal_(self.shared_tokens, std=0.02)
        self.shared_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.hr_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.rr_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.shared_ffn = self._make_ffn(feature_dim, hidden_dim, dropout)
        self.hr_ffn = self._make_ffn(feature_dim, hidden_dim, dropout)
        self.rr_ffn = self._make_ffn(feature_dim, hidden_dim, dropout)
        self.shared_gate = nn.Parameter(torch.tensor(-3.0))
        self.hr_gate = nn.Parameter(torch.tensor(-3.0))
        self.rr_gate = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _make_ffn(feature_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, shared_seq, hr_seq, rr_seq):
        batch_size = shared_seq.shape[0]
        tokens = self.shared_tokens.expand(batch_size, -1, -1)
        shared_memory, _ = self.shared_attn(
            self.shared_norm(tokens),
            self.shared_norm(shared_seq),
            self.shared_norm(shared_seq),
            need_weights=False,
        )
        shared_memory = tokens + self.shared_ffn(shared_memory)
        hr_delta, _ = self.hr_attn(self.hr_norm(hr_seq), shared_memory, shared_memory, need_weights=False)
        rr_delta, _ = self.rr_attn(self.rr_norm(rr_seq), shared_memory, shared_memory, need_weights=False)
        shared_delta, _ = self.shared_attn(
            self.shared_norm(shared_seq),
            self.shared_norm(torch.cat([hr_seq, rr_seq], dim=1)),
            self.shared_norm(torch.cat([hr_seq, rr_seq], dim=1)),
            need_weights=False,
        )
        shared_seq = shared_seq + torch.sigmoid(self.shared_gate) * self.shared_ffn(shared_delta)
        hr_seq = hr_seq + torch.sigmoid(self.hr_gate) * self.hr_ffn(hr_delta)
        rr_seq = rr_seq + torch.sigmoid(self.rr_gate) * self.rr_ffn(rr_delta)
        return shared_seq, hr_seq, rr_seq


class RROnlySlowAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1, bias=False)
        self.slow_blocks = nn.ModuleList()
        for kernel_size, dilation in ((15, 1), (31, 2), (45, 3), (61, 4)):
            padding = (kernel_size // 2) * dilation
            self.slow_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.slow_tokens = nn.Parameter(torch.zeros(1, 6, hidden_dim))
        nn.init.trunc_normal_(self.slow_tokens, std=0.02)
        self.slow_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.decode_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, rr_seq):
        x = self.input_proj(self.input_norm(rr_seq).permute(0, 2, 1))
        for block in self.slow_blocks:
            x = x + block(x)
        x_seq = x.permute(0, 2, 1)
        tokens = self.slow_tokens.expand(rr_seq.shape[0], -1, -1)
        tokens, _ = self.slow_attn(self.token_norm(tokens), self.token_norm(x_seq), self.token_norm(x_seq), need_weights=False)
        decoded, _ = self.decode_attn(self.token_norm(x_seq), self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        delta = self.output_proj(decoded.permute(0, 2, 1)).permute(0, 2, 1)
        return rr_seq + torch.sigmoid(self.residual_gate) * delta


class PrototypeScalarReadout(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_tokens=6, dropout=0.1):
        super().__init__()
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.prototype_tokens = nn.Parameter(torch.zeros(1, int(num_tokens), hidden_dim))
        nn.init.trunc_normal_(self.prototype_tokens, std=0.02)
        self.token_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.frame_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.frame_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            out,
        )

    def forward(self, z_seq):
        x = self.input_proj(self.input_norm(z_seq))
        tokens = self.prototype_tokens.expand(z_seq.shape[0], -1, -1)
        tokens_delta, _ = self.token_attn(tokens, x, x, need_weights=False)
        tokens = tokens + self.token_ffn(tokens_delta)
        refined, _ = self.frame_attn(x, tokens, tokens, need_weights=False)
        x = x + self.frame_ffn(refined)
        token_summary = torch.mean(tokens, dim=1)
        pooled = torch.cat(
            [
                token_summary,
                torch.mean(x, dim=1),
                torch.std(x, dim=1, unbiased=False),
                torch.amax(x, dim=1),
                torch.amin(x, dim=1),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(-1)


class PreTemporalPhysioAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_tokens=6, dropout=0.1):
        super().__init__()
        heads = 4 if hidden_dim % 4 == 0 else 1
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.branches = nn.ModuleList()
        for kernel_size, dilation in ((5, 1), (9, 2), (15, 4), (31, 4)):
            padding = (kernel_size // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.tokens = nn.Parameter(torch.zeros(1, int(num_tokens), hidden_dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.decode_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, dynamic_features):
        x = self.input_proj(self.input_norm(dynamic_features))
        y = x.permute(0, 2, 1)
        for branch in self.branches:
            y = y + branch(y)
        x = y.permute(0, 2, 1)
        tokens = self.tokens.expand(x.shape[0], -1, -1)
        tokens_delta, _ = self.token_attn(self.token_norm(tokens), self.token_norm(x), self.token_norm(x), need_weights=False)
        tokens = tokens + tokens_delta
        decoded, _ = self.decode_attn(self.token_norm(x), self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        delta = self.output_proj(decoded)
        return dynamic_features + torch.sigmoid(self.residual_gate) * delta


class RawMotionTemporalAdapter(nn.Module):
    def __init__(self, input_dim, stats_dim=10, hidden_dim=128, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.stats_norm = nn.LayerNorm(stats_dim)
        self.stats_proj = nn.Linear(stats_dim, hidden_dim)
        self.temporal_filter = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7, groups=hidden_dim, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=31, padding=15, groups=hidden_dim, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.trunc_normal_(self.output_proj[1].weight, std=1e-4)
        nn.init.zeros_(self.output_proj[1].bias)

    def forward(self, dynamic_features, motion_stats):
        state = self.stats_proj(self.stats_norm(motion_stats))
        filtered = self.temporal_filter(state.permute(0, 2, 1)).permute(0, 2, 1)
        delta = 0.01 * torch.tanh(self.output_proj(state + filtered))
        gate = torch.tanh(self.residual_gate)
        if torch.max(torch.abs(gate.detach())).item() < 1e-4:
            delta = delta.detach()
        return dynamic_features + gate * delta


class PhaseAwareTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        fs=30.0,
        low_bpm=45.0,
        high_bpm=150.0,
        num_freq_bins=96,
        dropout=0.1,
    ):
        super().__init__()
        self.fs = float(fs)
        self.low_bpm = float(low_bpm)
        self.high_bpm = float(high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.norm = nn.LayerNorm(hidden_dim)
        self.branches = nn.ModuleList()
        for kernel_size, dilation in [(7, 1), (11, 2), (15, 4), (21, 4), (31, 8)]:
            padding = (kernel_size // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                )
            )
        self.mix = nn.Sequential(
            nn.Conv1d(hidden_dim * len(self.branches), hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        self.spectral_gate = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.artifact_gate = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.phase_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.phase_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_gate = nn.Parameter(torch.tensor(0.2))

    def _band_power_and_hz(self, x):
        batch_size, frames, _ = x.shape
        centered = x - torch.mean(x, dim=1, keepdim=True)
        time = torch.arange(frames, device=x.device, dtype=x.dtype) / self.fs
        bpm = torch.linspace(
            self.low_bpm,
            self.high_bpm,
            self.num_freq_bins,
            device=x.device,
            dtype=x.dtype,
        )
        angles = 2.0 * torch.pi * (bpm[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bth,kt->bhk", centered, sin_basis)
        cos_score = torch.einsum("bth,kt->bhk", centered, cos_basis)
        power = (sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6
        weights = torch.softmax(torch.log(power), dim=1)
        hz = torch.sum(weights * (bpm / 60.0).unsqueeze(0), dim=1)
        return power, hz

    def _phase_tokens(self, hz, frames, dtype, device):
        time = torch.arange(frames, device=device, dtype=dtype)[None, :] / self.fs
        angles = 2.0 * torch.pi * hz[:, None].to(dtype=dtype) * time
        phase = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.phase_proj(phase)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        power, hz = self._band_power_and_hz(x_norm)
        y = x_norm.permute(0, 2, 1)
        y = self.mix(torch.cat([branch(y) for branch in self.branches], dim=1))
        band_gate = torch.sigmoid(self.spectral_gate(torch.log(power))).unsqueeze(-1)
        y = y * band_gate
        y = y * self.artifact_gate(x_norm.permute(0, 2, 1))
        phase = self._phase_tokens(hz, x.shape[1], x.dtype, x.device).permute(0, 2, 1)
        y = y + torch.tanh(self.phase_gate) * phase
        y = self.dropout(y).permute(0, 2, 1)
        return residual + torch.tanh(self.residual_gate) * y


class PhaseAwareTemporalMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        low_bpm=45.0,
        high_bpm=150.0,
        num_freq_bins=96,
        dropout=0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                PhaseAwareTemporalBlock(
                    hidden_dim=hidden_dim,
                    fs=fs,
                    low_bpm=low_bpm,
                    high_bpm=high_bpm,
                    num_freq_bins=num_freq_bins,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_size)
        self.skip_proj = nn.Linear(input_size, output_size)

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        for block in self.blocks:
            y = block(y)
        return self.output_proj(self.output_norm(y)) + self.skip_proj(x_norm)


class PhysioRepresentationMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        rr_low_bpm=6.0,
        rr_high_bpm=30.0,
        num_freq_bins=96,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        base_specs = [(5, 1), (7, 2), (9, 4), (11, 8)]
        long_specs = [(15, 1), (21, 2), (31, 4), (41, 8)]
        specs = long_specs if bool(long_context) else base_specs
        self.blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            kernel_size, dilation = specs[layer_idx % len(specs)]
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.hr_gate = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rr_gate = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_size)
        self.skip_proj = nn.Linear(input_size, output_size)
        self.band_gate = nn.Parameter(torch.tensor(0.0))
        self.attn_gate = nn.Parameter(torch.tensor(0.0))

    def _band_power(self, x, low_bpm, high_bpm):
        batch_size, frames, hidden_dim = x.shape
        centered = x - torch.mean(x, dim=1, keepdim=True)
        time = torch.arange(frames, device=x.device, dtype=x.dtype) / self.fs
        bpm = torch.linspace(float(low_bpm), float(high_bpm), self.num_freq_bins, device=x.device, dtype=x.dtype)
        angles = 2.0 * torch.pi * (bpm[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bth,kt->bhk", centered, sin_basis)
        cos_score = torch.einsum("bth,kt->bhk", centered, cos_basis)
        return torch.log((sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6)

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        y_conv = y.permute(0, 2, 1)
        for block in self.blocks:
            y_conv = y_conv + block(y_conv)
        y = y_conv.permute(0, 2, 1)
        hr_gate = torch.sigmoid(self.hr_gate(self._band_power(y, self.hr_low_bpm, self.hr_high_bpm))).unsqueeze(1)
        rr_gate = torch.sigmoid(self.rr_gate(self._band_power(y, self.rr_low_bpm, self.rr_high_bpm))).unsqueeze(1)
        y = y + torch.tanh(self.band_gate) * y * (hr_gate + rr_gate)
        attn_in = self.cross_norm(y)
        attn_out, _ = self.cross_attn(attn_in, attn_in, attn_in, need_weights=False)
        y = y + torch.tanh(self.attn_gate) * attn_out
        y = y + self.ffn(y)
        return self.output_proj(self.output_norm(y)) + self.skip_proj(x_norm)


class DualRatePhysioMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        rr_low_bpm=6.0,
        rr_high_bpm=30.0,
        num_freq_bins=96,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        self.input_norm = nn.LayerNorm(input_size)
        self.hr_input = nn.Linear(input_size, hidden_dim)
        self.rr_input = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1

        def make_blocks(specs):
            blocks = nn.ModuleList()
            for layer_idx in range(int(num_layers)):
                kernel_size, dilation = specs[layer_idx % len(specs)]
                padding = (kernel_size // 2) * dilation
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                        nn.GroupNorm(groups, hidden_dim),
                        nn.GELU(),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                        nn.GroupNorm(groups, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
            return blocks

        hr_specs = [(3, 1), (5, 1), (7, 2), (9, 2)]
        rr_specs = [(15, 2), (21, 4), (31, 6), (41, 8)] if bool(long_context) else [(9, 2), (15, 4), (21, 6), (31, 8)]
        self.hr_blocks = make_blocks(hr_specs)
        self.rr_blocks = make_blocks(rr_specs)
        self.hr_gate = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rr_gate = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.hr_norm = nn.LayerNorm(hidden_dim)
        self.rr_norm = nn.LayerNorm(hidden_dim)
        self.hr_to_rr = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.rr_to_hr = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.hr_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.rr_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.hr_output = nn.Linear(hidden_dim, output_size)
        self.rr_output = nn.Linear(hidden_dim, output_size)
        self.skip_proj = nn.Linear(input_size, output_size)
        self.hr_band_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_band_gate = nn.Parameter(torch.tensor(0.0))
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.last_hr_seq = None
        self.last_rr_seq = None

    def _band_power(self, x, low_bpm, high_bpm):
        batch_size, frames, hidden_dim = x.shape
        centered = x - torch.mean(x, dim=1, keepdim=True)
        time = torch.arange(frames, device=x.device, dtype=x.dtype) / self.fs
        bpm = torch.linspace(float(low_bpm), float(high_bpm), self.num_freq_bins, device=x.device, dtype=x.dtype)
        angles = 2.0 * torch.pi * (bpm[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bth,kt->bhk", centered, sin_basis)
        cos_score = torch.einsum("bth,kt->bhk", centered, cos_basis)
        return torch.log((sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6)

    def _run_blocks(self, x, blocks):
        y = x.permute(0, 2, 1)
        for block in blocks:
            y = y + block(y)
        return y.permute(0, 2, 1)

    def forward(self, x):
        x_norm = self.input_norm(x)
        hr_seq = self._run_blocks(self.hr_input(x_norm), self.hr_blocks)
        rr_seq = self._run_blocks(self.rr_input(x_norm), self.rr_blocks)
        hr_gate = torch.sigmoid(self.hr_gate(self._band_power(hr_seq, self.hr_low_bpm, self.hr_high_bpm))).unsqueeze(1)
        rr_gate = torch.sigmoid(self.rr_gate(self._band_power(rr_seq, self.rr_low_bpm, self.rr_high_bpm))).unsqueeze(1)
        hr_seq = hr_seq + torch.tanh(self.hr_band_gate) * hr_seq * hr_gate
        rr_seq = rr_seq + torch.tanh(self.rr_band_gate) * rr_seq * rr_gate
        hr_ctx, _ = self.rr_to_hr(self.hr_norm(hr_seq), self.rr_norm(rr_seq), self.rr_norm(rr_seq), need_weights=False)
        rr_ctx, _ = self.hr_to_rr(self.rr_norm(rr_seq), self.hr_norm(hr_seq), self.hr_norm(hr_seq), need_weights=False)
        cross_scale = torch.tanh(self.cross_gate)
        hr_seq = hr_seq + cross_scale * hr_ctx
        rr_seq = rr_seq + cross_scale * rr_ctx
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)
        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        fused = self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))
        return fused + self.skip_proj(x_norm)


class TaskTokenPhysioMixer(DualRatePhysioMixer):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        rr_low_bpm=6.0,
        rr_high_bpm=30.0,
        num_freq_bins=96,
        dropout=0.1,
        long_context=False,
        num_task_tokens=4,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fs=fs,
            hr_low_bpm=hr_low_bpm,
            hr_high_bpm=hr_high_bpm,
            rr_low_bpm=rr_low_bpm,
            rr_high_bpm=rr_high_bpm,
            num_freq_bins=num_freq_bins,
            dropout=dropout,
            long_context=long_context,
        )
        self.num_task_tokens = int(num_task_tokens)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.hr_tokens = nn.Parameter(torch.randn(1, self.num_task_tokens, hidden_dim) * 0.02)
        self.rr_tokens = nn.Parameter(torch.randn(1, self.num_task_tokens, hidden_dim) * 0.02)
        self.hr_token_norm = nn.LayerNorm(hidden_dim)
        self.rr_token_norm = nn.LayerNorm(hidden_dim)
        self.hr_token_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.rr_token_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.hr_frame_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.rr_frame_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.token_cross = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.hr_token_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.rr_token_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.token_gate = nn.Parameter(torch.tensor(0.0))
        self.frame_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_norm = self.input_norm(x)
        hr_seq = self._run_blocks(self.hr_input(x_norm), self.hr_blocks)
        rr_seq = self._run_blocks(self.rr_input(x_norm), self.rr_blocks)
        hr_gate = torch.sigmoid(self.hr_gate(self._band_power(hr_seq, self.hr_low_bpm, self.hr_high_bpm))).unsqueeze(1)
        rr_gate = torch.sigmoid(self.rr_gate(self._band_power(rr_seq, self.rr_low_bpm, self.rr_high_bpm))).unsqueeze(1)
        hr_seq = hr_seq + torch.tanh(self.hr_band_gate) * hr_seq * hr_gate
        rr_seq = rr_seq + torch.tanh(self.rr_band_gate) * rr_seq * rr_gate

        batch_size = x.shape[0]
        hr_tokens = self.hr_tokens.expand(batch_size, -1, -1)
        rr_tokens = self.rr_tokens.expand(batch_size, -1, -1)
        hr_token_ctx, _ = self.hr_token_attn(
            self.hr_token_norm(hr_tokens),
            self.hr_norm(hr_seq),
            self.hr_norm(hr_seq),
            need_weights=False,
        )
        rr_token_ctx, _ = self.rr_token_attn(
            self.rr_token_norm(rr_tokens),
            self.rr_norm(rr_seq),
            self.rr_norm(rr_seq),
            need_weights=False,
        )
        hr_tokens = hr_tokens + torch.tanh(self.token_gate) * hr_token_ctx
        rr_tokens = rr_tokens + torch.tanh(self.token_gate) * rr_token_ctx
        hr_cross, _ = self.token_cross(self.hr_token_norm(hr_tokens), self.rr_token_norm(rr_tokens), self.rr_token_norm(rr_tokens), need_weights=False)
        rr_cross, _ = self.token_cross(self.rr_token_norm(rr_tokens), self.hr_token_norm(hr_tokens), self.hr_token_norm(hr_tokens), need_weights=False)
        cross_scale = torch.tanh(self.cross_gate)
        hr_tokens = hr_tokens + cross_scale * hr_cross
        rr_tokens = rr_tokens + cross_scale * rr_cross
        hr_tokens = hr_tokens + self.hr_token_ffn(hr_tokens)
        rr_tokens = rr_tokens + self.rr_token_ffn(rr_tokens)

        hr_frame_ctx, _ = self.hr_frame_attn(self.hr_norm(hr_seq), self.hr_token_norm(hr_tokens), self.hr_token_norm(hr_tokens), need_weights=False)
        rr_frame_ctx, _ = self.rr_frame_attn(self.rr_norm(rr_seq), self.rr_token_norm(rr_tokens), self.rr_token_norm(rr_tokens), need_weights=False)
        frame_scale = torch.tanh(self.frame_gate)
        hr_seq = hr_seq + frame_scale * hr_frame_ctx
        rr_seq = rr_seq + frame_scale * rr_frame_ctx

        hr_ctx, _ = self.rr_to_hr(self.hr_norm(hr_seq), self.rr_norm(rr_seq), self.rr_norm(rr_seq), need_weights=False)
        rr_ctx, _ = self.hr_to_rr(self.rr_norm(rr_seq), self.hr_norm(hr_seq), self.hr_norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + cross_scale * hr_ctx
        rr_seq = rr_seq + cross_scale * rr_ctx
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)
        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        fused = self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))
        return fused + self.skip_proj(x_norm)


class PhysioOscillatorMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        rr_low_bpm=6.0,
        rr_high_bpm=30.0,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        specs = [(15, 1), (21, 2), (31, 4), (41, 8)] if bool(long_context) else [(5, 1), (9, 2), (15, 4), (21, 6)]
        self.blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            kernel_size, dilation = specs[layer_idx % len(specs)]
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.hr_delta = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.rr_delta = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.hr_amp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.rr_amp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.phase_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 6),
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.hr_output = nn.Linear(hidden_dim + 3, output_size)
        self.rr_output = nn.Linear(hidden_dim + 3, output_size)
        self.skip_proj = nn.Linear(input_size, output_size)
        self.last_hr_seq = None
        self.last_rr_seq = None
        self.last_oscillator_scalars = None

    def _bounded_delta(self, logits, low_bpm, high_bpm):
        low = float(low_bpm) / (60.0 * self.fs)
        high = float(high_bpm) / (60.0 * self.fs)
        return low + torch.sigmoid(logits).squeeze(-1) * (high - low)

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        y_conv = y.permute(0, 2, 1)
        for block in self.blocks:
            y_conv = y_conv + block(y_conv)
        y = y_conv.permute(0, 2, 1)

        hr_delta = self._bounded_delta(self.hr_delta(y), self.hr_low_bpm, self.hr_high_bpm)
        rr_delta = self._bounded_delta(self.rr_delta(y), self.rr_low_bpm, self.rr_high_bpm)
        hr_phase = torch.cumsum(hr_delta, dim=1)
        rr_phase = torch.cumsum(rr_delta, dim=1)
        hr_amp = torch.sigmoid(self.hr_amp(y)).squeeze(-1)
        rr_amp = torch.sigmoid(self.rr_amp(y)).squeeze(-1)
        hr_angle = 2.0 * torch.pi * hr_phase
        rr_angle = 2.0 * torch.pi * rr_phase
        hr_phase_features = torch.stack([torch.sin(hr_angle), torch.cos(hr_angle), hr_amp], dim=-1)
        rr_phase_features = torch.stack([torch.sin(rr_angle), torch.cos(rr_angle), rr_amp], dim=-1)
        phase_features = torch.cat([hr_phase_features, rr_phase_features], dim=-1)

        hr_bpm = torch.mean(hr_delta, dim=1) * self.fs * 60.0
        rr_bpm = torch.mean(rr_delta, dim=1) * self.fs * 60.0
        self.last_oscillator_scalars = {
            "hr": (hr_bpm - 80.0) / 30.0,
            "pr": (hr_bpm - 80.0) / 30.0,
            "rr": (rr_bpm - 16.0) / 8.0,
        }
        self.last_hr_seq = self.hr_output(torch.cat([y, hr_phase_features], dim=-1))
        self.last_rr_seq = self.rr_output(torch.cat([y, rr_phase_features], dim=-1))
        return self.phase_proj(torch.cat([y, phase_features], dim=-1)) + self.skip_proj(x_norm)


class RespEnvelopePhysioMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        cardio_specs = [(5, 1), (9, 2), (15, 4), (21, 6)]
        resp_specs = [(21, 4), (31, 8), (41, 12), (51, 16)] if bool(long_context) else [(15, 3), (21, 5), (31, 7), (41, 9)]
        self.cardio_blocks = nn.ModuleList()
        self.resp_blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            cardio_kernel, cardio_dilation = cardio_specs[layer_idx % len(cardio_specs)]
            resp_kernel, resp_dilation = resp_specs[layer_idx % len(resp_specs)]
            self.cardio_blocks.append(self._conv_block(hidden_dim, cardio_kernel, cardio_dilation, groups, dropout))
            self.resp_blocks.append(self._conv_block(hidden_dim, resp_kernel, resp_dilation, groups, dropout))
        self.envelope_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.cardio_to_resp = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.resp_to_cardio = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cardio_norm = nn.LayerNorm(hidden_dim)
        self.resp_norm = nn.LayerNorm(hidden_dim)
        self.cardio_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.resp_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.hr_output = nn.Linear(hidden_dim, output_size)
        self.rr_output = nn.Linear(hidden_dim, output_size)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.skip_proj = nn.Linear(input_size, output_size)
        self.last_hr_seq = None
        self.last_rr_seq = None

    @staticmethod
    def _conv_block(hidden_dim, kernel_size, dilation, groups, dropout):
        padding = (kernel_size // 2) * dilation
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        cardio = y.permute(0, 2, 1)
        resp = y.permute(0, 2, 1)
        for cardio_block, resp_block in zip(self.cardio_blocks, self.resp_blocks):
            cardio = cardio + cardio_block(cardio)
            resp = resp + resp_block(resp)
        cardio = cardio.permute(0, 2, 1)
        resp = resp.permute(0, 2, 1)

        envelope = self.envelope_gate(resp)
        cardio = cardio * (1.0 + envelope)
        resp_ctx, _ = self.cardio_to_resp(self.resp_norm(resp), self.cardio_norm(cardio), self.cardio_norm(cardio), need_weights=False)
        cardio_ctx, _ = self.resp_to_cardio(self.cardio_norm(cardio), self.resp_norm(resp), self.resp_norm(resp), need_weights=False)
        resp = resp + resp_ctx + self.resp_ffn(resp)
        cardio = cardio + cardio_ctx + self.cardio_ffn(cardio)

        self.last_hr_seq = self.hr_output(cardio)
        self.last_rr_seq = self.rr_output(resp)
        return self.fuse(torch.cat([cardio, resp], dim=-1)) + self.skip_proj(x_norm)


class SpectralTokenPhysioMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        rr_low_bpm=6.0,
        rr_high_bpm=30.0,
        num_freq_bins=48,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        specs = [(15, 1), (21, 2), (31, 4), (41, 8)] if bool(long_context) else [(5, 1), (9, 2), (15, 4), (21, 6)]
        self.blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            kernel_size, dilation = specs[layer_idx % len(specs)]
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.hr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        self.rr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        nn.init.trunc_normal_(self.hr_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.rr_freq_embed, std=0.02)
        self.freq_norm = nn.LayerNorm(hidden_dim)
        self.hr_token_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_token_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_frame_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_frame_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_norm = nn.LayerNorm(hidden_dim)
        self.rr_norm = nn.LayerNorm(hidden_dim)
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.hr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.rr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.hr_output = nn.Linear(hidden_dim, output_size)
        self.rr_output = nn.Linear(hidden_dim, output_size)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.skip_proj = nn.Linear(input_size, output_size)
        self.last_hr_seq = None
        self.last_rr_seq = None

    def _band_tokens(self, mag, freq_bpm, low_bpm, high_bpm, embed):
        mask = (freq_bpm >= float(low_bpm)) & (freq_bpm <= float(high_bpm))
        if torch.count_nonzero(mask).item() == 0:
            mask = freq_bpm > 0
        band = mag[:, mask, :]
        if band.shape[1] == 0:
            band = mag
        band = F.interpolate(
            band.permute(0, 2, 1),
            size=self.num_freq_bins,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return self.freq_norm(band) + embed.to(device=band.device, dtype=band.dtype)

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        y_conv = y.permute(0, 2, 1)
        for block in self.blocks:
            y_conv = y_conv + block(y_conv)
        y = y_conv.permute(0, 2, 1)

        spec = torch.fft.rfft(y.float(), dim=1, norm="ortho")
        mag = torch.log1p(torch.abs(spec)).to(dtype=y.dtype)
        freq_hz = torch.fft.rfftfreq(y.shape[1], d=1.0 / self.fs).to(device=y.device)
        freq_bpm = freq_hz * 60.0
        hr_tokens = self._band_tokens(mag, freq_bpm, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(mag, freq_bpm, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_tokens, _ = self.hr_token_attn(self.freq_norm(hr_tokens), self.freq_norm(hr_tokens), self.freq_norm(hr_tokens), need_weights=False)
        rr_tokens, _ = self.rr_token_attn(self.freq_norm(rr_tokens), self.freq_norm(rr_tokens), self.freq_norm(rr_tokens), need_weights=False)

        frame = self.frame_norm(y)
        hr_ctx, _ = self.hr_frame_attn(frame, self.freq_norm(hr_tokens), self.freq_norm(hr_tokens), need_weights=False)
        rr_ctx, _ = self.rr_frame_attn(frame, self.freq_norm(rr_tokens), self.freq_norm(rr_tokens), need_weights=False)
        cross = torch.tanh(self.cross_gate)
        hr_seq = y + hr_ctx + cross * rr_ctx
        rr_seq = y + rr_ctx + cross * hr_ctx
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        return self.fuse(torch.cat([hr_seq, rr_seq], dim=-1)) + self.skip_proj(x_norm)


class LongContextSpectralMemoryMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=180.0,
        rr_low_bpm=6.0,
        rr_high_bpm=45.0,
        num_freq_bins=48,
        segment_frames=160,
        dropout=0.1,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        self.segment_frames = int(segment_frames)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.local_blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            kernel_size = [5, 9, 15, 21][layer_idx % 4]
            dilation = [1, 2, 4, 6][layer_idx % 4]
            padding = (kernel_size // 2) * dilation
            self.local_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.segment_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.hr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        self.rr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        nn.init.trunc_normal_(self.hr_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.rr_freq_embed, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.memory_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_freq_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_freq_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.rr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.hr_output = nn.Linear(hidden_dim, output_size)
        self.rr_output = nn.Linear(hidden_dim, output_size)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.skip_proj = nn.Linear(input_size, output_size)
        self.last_hr_seq = None
        self.last_rr_seq = None

    def _band_tokens(self, y, low_bpm, high_bpm, embed):
        spec = torch.fft.rfft(y.float(), dim=1, norm="ortho")
        mag = torch.log1p(torch.abs(spec)).to(dtype=y.dtype)
        freq_hz = torch.fft.rfftfreq(y.shape[1], d=1.0 / self.fs).to(device=y.device)
        freq_bpm = freq_hz * 60.0
        mask = (freq_bpm >= float(low_bpm)) & (freq_bpm <= float(high_bpm))
        if torch.count_nonzero(mask).item() == 0:
            mask = freq_bpm > 0
        band = mag[:, mask, :]
        if band.shape[1] == 0:
            band = mag
        band = F.interpolate(
            band.permute(0, 2, 1),
            size=self.num_freq_bins,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return self.norm(band) + embed.to(device=band.device, dtype=band.dtype)

    def _memory_tokens(self, y):
        batch, frames, channels = y.shape
        segment_frames = max(self.segment_frames, 1)
        segments = int(torch.ceil(torch.tensor(frames / segment_frames)).item())
        padded_frames = segments * segment_frames
        if padded_frames > frames:
            pad = y[:, -1:, :].expand(batch, padded_frames - frames, channels)
            y_pad = torch.cat([y, pad], dim=1)
        else:
            y_pad = y
        y_seg = y_pad.reshape(batch, segments, segment_frames, channels)
        tokens = torch.cat(
            [
                torch.mean(y_seg, dim=2),
                torch.std(y_seg, dim=2, unbiased=False),
            ],
            dim=-1,
        )
        return self.memory_encoder(self.segment_proj(tokens))

    def forward(self, x):
        x_norm = self.input_norm(x)
        y = self.input_proj(x_norm)
        y_conv = y.permute(0, 2, 1)
        for block in self.local_blocks:
            y_conv = y_conv + block(y_conv)
        y = y_conv.permute(0, 2, 1)

        memory_tokens = self._memory_tokens(y)
        hr_tokens = self._band_tokens(y, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(y, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        memory_ctx, _ = self.memory_attn(self.norm(y), self.norm(memory_tokens), self.norm(memory_tokens), need_weights=False)
        hr_ctx, _ = self.hr_freq_attn(self.norm(y + memory_ctx), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_ctx, _ = self.rr_freq_attn(self.norm(y + memory_ctx), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = y + memory_ctx + hr_ctx
        rr_seq = y + memory_ctx + rr_ctx
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)
        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        return self.fuse(torch.cat([hr_seq, rr_seq], dim=-1)) + self.skip_proj(x_norm)


class DualBandCrossAttentionMixer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim=128,
        num_layers=4,
        fs=30.0,
        hr_low_bpm=45.0,
        hr_high_bpm=180.0,
        rr_low_bpm=6.0,
        rr_high_bpm=45.0,
        num_freq_bins=48,
        dropout=0.1,
        long_context=False,
    ):
        super().__init__()
        self.fs = float(fs)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.rr_low_bpm = float(rr_low_bpm)
        self.rr_high_bpm = float(rr_high_bpm)
        self.num_freq_bins = int(num_freq_bins)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        groups = 8 if hidden_dim % 8 == 0 else 1
        shared_specs = [(5, 1), (9, 2), (15, 4), (21, 6)]
        hr_specs = [(5, 1), (7, 2), (9, 3), (11, 4)]
        rr_specs = [(15, 2), (21, 4), (31, 8), (41, 12)] if bool(long_context) else [(11, 2), (15, 4), (21, 6), (31, 8)]
        self.shared_blocks = nn.ModuleList()
        self.hr_blocks = nn.ModuleList()
        self.rr_blocks = nn.ModuleList()
        for layer_idx in range(int(num_layers)):
            self.shared_blocks.append(self._conv_block(hidden_dim, groups, *shared_specs[layer_idx % len(shared_specs)], dropout))
            self.hr_blocks.append(self._conv_block(hidden_dim, groups, *hr_specs[layer_idx % len(hr_specs)], dropout))
            self.rr_blocks.append(self._conv_block(hidden_dim, groups, *rr_specs[layer_idx % len(rr_specs)], dropout))
        self.hr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        self.rr_freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        nn.init.trunc_normal_(self.hr_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.rr_freq_embed, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hr_band_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_band_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_from_rr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_from_hr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_cross_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_cross_gate = nn.Parameter(torch.tensor(0.0))
        self.hr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.rr_ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.hr_output = nn.Linear(hidden_dim, output_size)
        self.rr_output = nn.Linear(hidden_dim, output_size)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.skip_proj = nn.Linear(input_size, output_size)
        self.last_hr_seq = None
        self.last_rr_seq = None

    @staticmethod
    def _conv_block(channels, groups, kernel_size, dilation, dropout):
        padding = (kernel_size // 2) * dilation
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _band_tokens(self, y, low_bpm, high_bpm, embed):
        spec = torch.fft.rfft(y.float(), dim=1, norm="ortho")
        mag = torch.log1p(torch.abs(spec)).to(dtype=y.dtype)
        freq_hz = torch.fft.rfftfreq(y.shape[1], d=1.0 / self.fs).to(device=y.device)
        freq_bpm = freq_hz * 60.0
        mask = (freq_bpm >= float(low_bpm)) & (freq_bpm <= float(high_bpm))
        if torch.count_nonzero(mask).item() == 0:
            mask = freq_bpm > 0
        band = mag[:, mask, :]
        if band.shape[1] == 0:
            band = mag
        band = F.interpolate(
            band.permute(0, 2, 1),
            size=self.num_freq_bins,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return self.norm(band) + embed.to(device=band.device, dtype=band.dtype)

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class ComplexDualBandCrossAttentionMixer(DualBandCrossAttentionMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        self.complex_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _band_tokens(self, y, low_bpm, high_bpm, embed):
        spec = torch.fft.rfft(y.float(), dim=1, norm="ortho")
        mag = torch.log1p(torch.abs(spec)).to(dtype=y.dtype)
        phase = torch.angle(spec).to(dtype=y.dtype)
        phase_tokens = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=-1)
        freq_hz = torch.fft.rfftfreq(y.shape[1], d=1.0 / self.fs).to(device=y.device)
        freq_bpm = freq_hz * 60.0
        mask = (freq_bpm >= float(low_bpm)) & (freq_bpm <= float(high_bpm))
        if torch.count_nonzero(mask).item() == 0:
            mask = freq_bpm > 0
        band = phase_tokens[:, mask, :]
        if band.shape[1] == 0:
            band = phase_tokens
        band = F.interpolate(
            band.permute(0, 2, 1),
            size=self.num_freq_bins,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        band = self.complex_token_proj(band)
        return self.norm(band) + embed.to(device=band.device, dtype=band.dtype)


class ComplexPhysioPrototypeMemoryMixer(ComplexDualBandCrossAttentionMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        proto_bins = self.num_freq_bins
        hr_freqs = torch.linspace(self.hr_low_bpm / 60.0, self.hr_high_bpm / 60.0, proto_bins)
        rr_freqs = torch.linspace(self.rr_low_bpm / 60.0, self.rr_high_bpm / 60.0, proto_bins)
        self.register_buffer("hr_proto_freqs_hz", hr_freqs, persistent=False)
        self.register_buffer("rr_proto_freqs_hz", rr_freqs, persistent=False)
        self.hr_proto_embed = nn.Parameter(torch.zeros(1, proto_bins, hidden_dim))
        self.rr_proto_embed = nn.Parameter(torch.zeros(1, proto_bins, hidden_dim))
        nn.init.trunc_normal_(self.hr_proto_embed, std=0.02)
        nn.init.trunc_normal_(self.rr_proto_embed, std=0.02)
        self.hr_proto_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=kwargs.get("dropout", 0.1), batch_first=True)
        self.rr_proto_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=kwargs.get("dropout", 0.1), batch_first=True)
        self.hr_proto_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_proto_gate = nn.Parameter(torch.tensor(0.0))

    def _prototype_tokens(self, seq, freqs_hz, embed):
        seq_norm = self.norm(seq)
        time = torch.arange(seq.shape[1], device=seq.device, dtype=torch.float32) / float(self.fs)
        phase = 2.0 * math.pi * time[:, None] * freqs_hz.to(device=seq.device, dtype=torch.float32)[None, :]
        sin_basis = torch.sin(phase).to(dtype=seq.dtype)
        cos_basis = torch.cos(phase).to(dtype=seq.dtype)
        scale = max(float(seq.shape[1]), 1.0) ** 0.5
        sin_coeff = torch.einsum("bth,tp->bph", seq_norm, sin_basis) / scale
        cos_coeff = torch.einsum("bth,tp->bph", seq_norm, cos_basis) / scale
        tokens = torch.sqrt(sin_coeff.square() + cos_coeff.square() + 1e-6)
        return self.norm(tokens) + embed.to(device=seq.device, dtype=seq.dtype)

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class DecoupledPhysioPrototypeMemoryMixer(ComplexPhysioPrototypeMemoryMixer):
    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class OrthogonalizedPhysioPrototypeMemoryMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        self.nuisance_tokens = nn.Parameter(torch.zeros(1, self.num_freq_bins, hidden_dim))
        nn.init.trunc_normal_(self.nuisance_tokens, std=0.02)
        self.nuisance_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.hr_nuisance_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_nuisance_gate = nn.Parameter(torch.tensor(-2.0))
        self.nuisance_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _remove_nuisance(self, seq, nuisance, gate):
        seq_norm = self.norm(seq)
        nuisance_norm = self.norm(nuisance)
        scale = float(seq_norm.shape[-1]) ** -0.5
        weights = torch.softmax(torch.matmul(seq_norm, nuisance_norm.transpose(1, 2)) * scale, dim=-1)
        nuisance_frame = torch.matmul(weights, nuisance_norm)
        projection = (seq_norm * nuisance_frame).sum(dim=-1, keepdim=True) / (
            nuisance_frame.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        return seq - torch.sigmoid(gate) * projection * nuisance_frame

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        nuisance_query = self.nuisance_tokens.to(device=shared.device, dtype=shared.dtype).expand(shared.shape[0], -1, -1)
        nuisance, _ = self.nuisance_attn(self.norm(nuisance_query), self.norm(shared), self.norm(shared), need_weights=False)
        nuisance = nuisance_query + nuisance + self.nuisance_ffn(nuisance)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        hr_seq = self._remove_nuisance(hr_seq, nuisance, self.hr_nuisance_gate)
        rr_seq = self._remove_nuisance(rr_seq, nuisance, self.rr_nuisance_gate)

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class CrossScaleCardioRespDecompositionMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.cross_scale_slow_factor = 4
        self.cross_scale_hr_fast = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_hr_carrier = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False
        )
        self.cross_scale_rr_slow_gru = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.cross_scale_env_slow_gru = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.cross_scale_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_rr_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_hr_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_rr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_hr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_hr_gate = nn.Parameter(torch.tensor(-2.5))
        self.cross_scale_rr_gate = nn.Parameter(torch.tensor(-2.5))
        self.cross_scale_out_gate = nn.Parameter(torch.tensor(-3.0))

    def _cross_scale_slow(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.cross_scale_slow_factor,
            stride=self.cross_scale_slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.norm(slow))
        slow_ctx, _ = self.cross_scale_slow_self(self.norm(slow), self.norm(slow), self.norm(slow), need_weights=False)
        return slow + slow_ctx

    def _cross_scale_envelope(self, hr_seq):
        carrier = self.cross_scale_hr_carrier(hr_seq.permute(0, 2, 1)).permute(0, 2, 1)
        centered = carrier - torch.mean(carrier, dim=1, keepdim=True)
        return self.cross_scale_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_fast = self.cross_scale_hr_fast(hr_seq)
        hr_env = self._cross_scale_envelope(hr_seq)
        rr_slow = self._cross_scale_slow(rr_seq, self.cross_scale_rr_slow_gru)
        env_slow = self._cross_scale_slow(hr_env, self.cross_scale_env_slow_gru)
        rr_slow_ctx, _ = self.cross_scale_rr_decode(self.norm(rr_seq), self.norm(rr_slow), self.norm(rr_slow), need_weights=False)
        rr_env_ctx, _ = self.cross_scale_env_decode(self.norm(rr_seq), self.norm(env_slow), self.norm(env_slow), need_weights=False)
        hr_from_rr, _ = self.cross_scale_hr_decode(self.norm(hr_seq), self.norm(rr_slow), self.norm(rr_slow), need_weights=False)
        rr_delta = self.cross_scale_rr_update(torch.cat([rr_slow_ctx, rr_env_ctx], dim=-1))
        hr_delta = self.cross_scale_hr_update(torch.cat([hr_fast, hr_from_rr], dim=-1))
        hr_seq = hr_seq + torch.sigmoid(self.cross_scale_hr_gate) * hr_delta
        rr_seq = rr_seq + torch.sigmoid(self.cross_scale_rr_gate) * rr_delta
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.cross_scale_out_gate) * self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))


class SubharmonicCardioRespMemoryMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        proto_bins = self.num_freq_bins
        self.hr_subharmonic_embed = nn.Parameter(torch.zeros(1, proto_bins, hidden_dim))
        nn.init.trunc_normal_(self.hr_subharmonic_embed, std=0.02)
        self.subharmonic_to_rr = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rr_to_carrier = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.carrier_modulation = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.subharmonic_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.subharmonic_rr_gate = nn.Parameter(torch.tensor(0.0))
        self.carrier_mod_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        subharmonic_tokens = self._prototype_tokens(hr_seq, self.rr_proto_freqs_hz, self.hr_subharmonic_embed)
        rr_subharmonic, _ = self.subharmonic_to_rr(
            self.norm(rr_seq),
            self.norm(subharmonic_tokens),
            self.norm(subharmonic_tokens),
            need_weights=False,
        )
        rr_seq = rr_seq + torch.tanh(self.subharmonic_rr_gate) * self.subharmonic_ffn(rr_subharmonic)
        carrier_ctx, _ = self.rr_to_carrier(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        hr_seq = hr_seq * (1.0 + torch.tanh(self.carrier_mod_gate) * self.carrier_modulation(carrier_ctx))

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class LowFreqMotionStateSubharmonicMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        input_size = int(kwargs.get("input_size", args[0] if args else 0))
        output_size = int(kwargs.get("output_size", kwargs.get("feature_dim", 128)))
        dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(*args, **kwargs)
        self.motion_input = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.motion_low_15 = nn.Conv1d(output_size, output_size, kernel_size=15, padding=7, groups=output_size, bias=False)
        self.motion_low_31 = nn.Conv1d(output_size, output_size, kernel_size=31, padding=15, groups=output_size, bias=False)
        self.motion_state_norm = nn.LayerNorm(output_size)
        self.motion_state_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.motion_rr_update = nn.Sequential(
            nn.LayerNorm(output_size * 3),
            nn.Linear(output_size * 3, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.motion_hr_update = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.motion_rr_gate = nn.Parameter(torch.tensor(0.0))
        self.motion_hr_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.motion_rr_update[-1].weight)
        nn.init.zeros_(self.motion_rr_update[-1].bias)
        nn.init.zeros_(self.motion_hr_update[-1].weight)
        nn.init.zeros_(self.motion_hr_update[-1].bias)

    def _motion_state(self, x):
        motion = self.motion_input(x)
        motion_t = motion.permute(0, 2, 1)
        smooth = F.avg_pool1d(motion_t, kernel_size=9, stride=1, padding=4)
        low = smooth + self.motion_low_15(motion_t) + self.motion_low_31(motion_t)
        state, _ = self.motion_state_gru(self.motion_state_norm(low.permute(0, 2, 1)))
        return state

    def forward(self, x):
        base = super().forward(x)
        state = self._motion_state(x)
        centered_state = state - torch.mean(state, dim=1, keepdim=True)
        rr_delta = self.motion_rr_update(torch.cat([self.last_rr_seq, state, centered_state.abs()], dim=-1))
        hr_delta = self.motion_hr_update(torch.cat([self.last_hr_seq, state], dim=-1))
        self.last_rr_seq = self.last_rr_seq + torch.tanh(self.motion_rr_gate) * rr_delta
        self.last_hr_seq = self.last_hr_seq + torch.tanh(self.motion_hr_gate) * hr_delta
        return base


class SubharmonicRateAnchorMemoryMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        num_anchors = min(12, max(4, self.num_freq_bins // 4))
        self.rate_anchor_hr = nn.Parameter(torch.zeros(1, num_anchors, hidden_dim))
        self.rate_anchor_rr = nn.Parameter(torch.zeros(1, num_anchors, hidden_dim))
        nn.init.trunc_normal_(self.rate_anchor_hr, std=0.02)
        nn.init.trunc_normal_(self.rate_anchor_rr, std=0.02)
        self.rate_anchor_norm = nn.LayerNorm(hidden_dim)
        self.rate_anchor_hr_pool = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_rr_pool = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_hr_from_rr = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_rr_from_hr = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_hr_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_rr_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rate_anchor_hr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rate_anchor_rr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rate_anchor_hr_gate = nn.Parameter(torch.tensor(0.0))
        self.rate_anchor_rr_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.trunc_normal_(self.rate_anchor_hr_update[-1].weight, std=1e-4)
        nn.init.zeros_(self.rate_anchor_hr_update[-1].bias)
        nn.init.trunc_normal_(self.rate_anchor_rr_update[-1].weight, std=1e-4)
        nn.init.zeros_(self.rate_anchor_rr_update[-1].bias)

    def forward(self, x):
        base = super().forward(x)
        batch_size = self.last_hr_seq.shape[0]
        hr_anchor = self.rate_anchor_hr.expand(batch_size, -1, -1)
        rr_anchor = self.rate_anchor_rr.expand(batch_size, -1, -1)
        hr_anchor, _ = self.rate_anchor_hr_pool(
            self.rate_anchor_norm(hr_anchor),
            self.rate_anchor_norm(self.last_hr_seq),
            self.rate_anchor_norm(self.last_hr_seq),
            need_weights=False,
        )
        rr_anchor, _ = self.rate_anchor_rr_pool(
            self.rate_anchor_norm(rr_anchor),
            self.rate_anchor_norm(self.last_rr_seq),
            self.rate_anchor_norm(self.last_rr_seq),
            need_weights=False,
        )
        hr_ctx, _ = self.rate_anchor_hr_from_rr(
            self.rate_anchor_norm(hr_anchor),
            self.rate_anchor_norm(rr_anchor),
            self.rate_anchor_norm(rr_anchor),
            need_weights=False,
        )
        rr_ctx, _ = self.rate_anchor_rr_from_hr(
            self.rate_anchor_norm(rr_anchor),
            self.rate_anchor_norm(hr_anchor),
            self.rate_anchor_norm(hr_anchor),
            need_weights=False,
        )
        hr_anchor = hr_anchor + hr_ctx
        rr_anchor = rr_anchor + rr_ctx
        hr_delta, _ = self.rate_anchor_hr_decode(
            self.rate_anchor_norm(self.last_hr_seq),
            self.rate_anchor_norm(hr_anchor),
            self.rate_anchor_norm(hr_anchor),
            need_weights=False,
        )
        rr_delta, _ = self.rate_anchor_rr_decode(
            self.rate_anchor_norm(self.last_rr_seq),
            self.rate_anchor_norm(rr_anchor),
            self.rate_anchor_norm(rr_anchor),
            need_weights=False,
        )
        self.last_hr_seq = self.last_hr_seq + torch.tanh(self.rate_anchor_hr_gate) * self.rate_anchor_hr_update(hr_delta)
        self.last_rr_seq = self.last_rr_seq + torch.tanh(self.rate_anchor_rr_gate) * self.rate_anchor_rr_update(rr_delta)
        return base


class SubharmonicCrossScaleMemoryMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.cross_scale_slow_factor = 4
        self.cross_scale_hr_carrier = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False
        )
        self.cross_scale_rr_slow_gru = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.cross_scale_env_slow_gru = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.cross_scale_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_rr_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_scale_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_rr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_scale_rr_gate = nn.Parameter(torch.tensor(-2.5))
        self.cross_scale_out_gate = nn.Parameter(torch.tensor(-3.0))

    def _cross_scale_slow(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.cross_scale_slow_factor,
            stride=self.cross_scale_slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.norm(slow))
        slow_ctx, _ = self.cross_scale_slow_self(self.norm(slow), self.norm(slow), self.norm(slow), need_weights=False)
        return slow + slow_ctx

    def _cross_scale_envelope(self, hr_seq):
        carrier = self.cross_scale_hr_carrier(hr_seq.permute(0, 2, 1)).permute(0, 2, 1)
        centered = carrier - torch.mean(carrier, dim=1, keepdim=True)
        return self.cross_scale_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._cross_scale_slow(rr_seq, self.cross_scale_rr_slow_gru)
        env_slow = self._cross_scale_slow(self._cross_scale_envelope(hr_seq), self.cross_scale_env_slow_gru)
        rr_slow_ctx, _ = self.cross_scale_rr_decode(self.norm(rr_seq), self.norm(rr_slow), self.norm(rr_slow), need_weights=False)
        rr_env_ctx, _ = self.cross_scale_env_decode(self.norm(rr_seq), self.norm(env_slow), self.norm(env_slow), need_weights=False)
        rr_delta = self.cross_scale_rr_update(torch.cat([rr_slow_ctx, rr_env_ctx], dim=-1))
        rr_seq = rr_seq + torch.sigmoid(self.cross_scale_rr_gate) * rr_delta
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.cross_scale_out_gate) * self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))


class SubharmonicCrossScaleReadoutMixer(SubharmonicCrossScaleMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_scale_rr_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        base = SubharmonicCardioRespMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._cross_scale_slow(rr_seq, self.cross_scale_rr_slow_gru)
        env_slow = self._cross_scale_slow(self._cross_scale_envelope(hr_seq), self.cross_scale_env_slow_gru)
        rr_slow_ctx, _ = self.cross_scale_rr_decode(self.norm(rr_seq), self.norm(rr_slow), self.norm(rr_slow), need_weights=False)
        rr_env_ctx, _ = self.cross_scale_env_decode(self.norm(rr_seq), self.norm(env_slow), self.norm(env_slow), need_weights=False)
        rr_delta = self.cross_scale_rr_update(torch.cat([rr_slow_ctx, rr_env_ctx], dim=-1))
        self.last_rr_seq = rr_seq + torch.sigmoid(self.cross_scale_rr_gate) * rr_delta
        return base


class SubharmonicEnvelopeResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.envelope_filter = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=15, padding=7, groups=hidden_dim, bias=False
        )
        self.envelope_norm = nn.LayerNorm(hidden_dim)
        self.envelope_to_rr = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rr_to_envelope = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.envelope_rr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.envelope_hr_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.envelope_rr_gate = nn.Parameter(torch.tensor(-3.0))
        self.envelope_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.envelope_out_gate = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.envelope_rr_update[-1].weight)
        nn.init.zeros_(self.envelope_rr_update[-1].bias)
        nn.init.zeros_(self.envelope_hr_update[-1].weight)
        nn.init.zeros_(self.envelope_hr_update[-1].bias)

    def _hr_envelope(self, hr_seq):
        carrier = self.envelope_filter(hr_seq.permute(0, 2, 1)).permute(0, 2, 1)
        centered = carrier - carrier.mean(dim=1, keepdim=True)
        envelope = centered.abs() + 0.5 * centered.square()
        return self.envelope_norm(envelope)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        envelope = self._hr_envelope(hr_seq)
        rr_from_env, _ = self.envelope_to_rr(self.norm(rr_seq), envelope, envelope, need_weights=False)
        env_from_rr, _ = self.rr_to_envelope(envelope, self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_delta = self.envelope_rr_update(torch.cat([rr_seq, rr_from_env], dim=-1))
        hr_delta = self.envelope_hr_update(torch.cat([hr_seq, env_from_rr], dim=-1))
        rr_seq = rr_seq + torch.sigmoid(self.envelope_rr_gate) * rr_delta
        hr_seq = hr_seq + torch.sigmoid(self.envelope_hr_gate) * hr_delta
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.envelope_out_gate) * self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))


class SubharmonicEnvelopeRRReadoutMixer(SubharmonicEnvelopeResidualMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.envelope_rr_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        base = SubharmonicCardioRespMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        envelope = self._hr_envelope(hr_seq)
        rr_from_env, _ = self.envelope_to_rr(self.norm(rr_seq), envelope, envelope, need_weights=False)
        rr_delta = self.envelope_rr_update(torch.cat([rr_seq, rr_from_env], dim=-1))
        self.last_rr_seq = rr_seq + torch.sigmoid(self.envelope_rr_gate) * rr_delta
        return base


class SubharmonicInstanceStableResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        stat_dim = int(output_size) * 6
        self.instance_stable_norm = nn.LayerNorm(stat_dim)
        self.instance_stable_mlp = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(output_size) * 2),
        )
        self.instance_stable_hr_update = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(output_size), int(output_size)),
        )
        self.instance_stable_rr_update = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(output_size), int(output_size)),
        )
        self.instance_stable_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.instance_stable_rr_gate = nn.Parameter(torch.tensor(-3.5))
        nn.init.zeros_(self.instance_stable_mlp[-1].weight)
        nn.init.zeros_(self.instance_stable_mlp[-1].bias)
        nn.init.zeros_(self.instance_stable_hr_update[-1].weight)
        nn.init.zeros_(self.instance_stable_hr_update[-1].bias)
        nn.init.zeros_(self.instance_stable_rr_update[-1].weight)
        nn.init.zeros_(self.instance_stable_rr_update[-1].bias)

    @staticmethod
    def _stream_stats(seq):
        mean = seq.mean(dim=1)
        std = seq.std(dim=1, unbiased=False)
        span = seq.amax(dim=1) - seq.amin(dim=1)
        return mean, std, span

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        stats = torch.cat([*self._stream_stats(hr_seq), *self._stream_stats(rr_seq)], dim=-1)
        affine = self.instance_stable_mlp(self.instance_stable_norm(stats))
        hr_shift, rr_shift = affine.chunk(2, dim=-1)
        hr_delta = self.instance_stable_hr_update(hr_seq) + hr_shift.unsqueeze(1)
        rr_delta = self.instance_stable_rr_update(rr_seq) + rr_shift.unsqueeze(1)
        self.last_hr_seq = hr_seq + torch.sigmoid(self.instance_stable_hr_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.instance_stable_rr_gate) * rr_delta
        return base


class SubharmonicStreamNormResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        self.stream_stable_hr_update = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(output_size), int(output_size)),
        )
        self.stream_stable_rr_update = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(output_size), int(output_size)),
        )
        self.stream_stable_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.stream_stable_rr_gate = nn.Parameter(torch.tensor(-3.5))
        nn.init.zeros_(self.stream_stable_hr_update[-1].weight)
        nn.init.zeros_(self.stream_stable_hr_update[-1].bias)
        nn.init.zeros_(self.stream_stable_rr_update[-1].weight)
        nn.init.zeros_(self.stream_stable_rr_update[-1].bias)

    @staticmethod
    def _instance_norm(seq):
        mean = seq.mean(dim=1, keepdim=True)
        std = seq.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (seq - mean) / std

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_delta = self.stream_stable_hr_update(self._instance_norm(hr_seq))
        rr_delta = self.stream_stable_rr_update(self._instance_norm(rr_seq))
        self.last_hr_seq = hr_seq + torch.sigmoid(self.stream_stable_hr_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.stream_stable_rr_gate) * rr_delta
        return base


class SubharmonicTaskBasisResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        rank = 8
        self.task_basis_norm = nn.LayerNorm(int(output_size))
        self.hr_task_basis = nn.Parameter(torch.zeros(rank, int(output_size)))
        self.rr_task_basis = nn.Parameter(torch.zeros(rank, int(output_size)))
        nn.init.trunc_normal_(self.hr_task_basis, std=0.02)
        nn.init.trunc_normal_(self.rr_task_basis, std=0.02)
        self.hr_task_coeff = nn.Sequential(
            nn.Linear(int(output_size), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank),
        )
        self.rr_task_coeff = nn.Sequential(
            nn.Linear(int(output_size), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank),
        )
        self.hr_task_gate = nn.Parameter(torch.tensor(-4.0))
        self.rr_task_gate = nn.Parameter(torch.tensor(-3.5))
        nn.init.zeros_(self.hr_task_coeff[-1].weight)
        nn.init.zeros_(self.hr_task_coeff[-1].bias)
        nn.init.zeros_(self.rr_task_coeff[-1].weight)
        nn.init.zeros_(self.rr_task_coeff[-1].bias)

    def _basis_residual(self, seq, coeff_head, basis):
        coeff = torch.tanh(coeff_head(self.task_basis_norm(seq)))
        return torch.einsum("btr,rf->btf", coeff, basis)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_delta = self._basis_residual(hr_seq, self.hr_task_coeff, self.hr_task_basis)
        rr_delta = self._basis_residual(rr_seq, self.rr_task_coeff, self.rr_task_basis)
        self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_task_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.rr_task_gate) * rr_delta
        return base


class SubharmonicChannelCalibrationMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        self.channel_calib_hr_scale = nn.Parameter(torch.zeros(1, 1, int(output_size)))
        self.channel_calib_rr_scale = nn.Parameter(torch.zeros(1, 1, int(output_size)))
        self.channel_calib_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.channel_calib_rr_gate = nn.Parameter(torch.tensor(-3.5))

    def forward(self, x):
        base = super().forward(x)
        hr_gain = 1.0 + torch.sigmoid(self.channel_calib_hr_gate) * torch.tanh(self.channel_calib_hr_scale)
        rr_gain = 1.0 + torch.sigmoid(self.channel_calib_rr_gate) * torch.tanh(self.channel_calib_rr_scale)
        self.last_hr_seq = self.last_hr_seq * hr_gain
        self.last_rr_seq = self.last_rr_seq * rr_gain
        return base


class SubharmonicStateResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = min(64, int(output_size))
        dropout = kwargs.get("dropout", 0.1)
        self.state_residual_hr_norm = nn.LayerNorm(int(output_size))
        self.state_residual_rr_norm = nn.LayerNorm(int(output_size))
        self.state_residual_hr_gru = nn.GRU(
            int(output_size), hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.state_residual_rr_gru = nn.GRU(
            int(output_size), hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.state_residual_hr_out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, int(output_size)),
        )
        self.state_residual_rr_out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, int(output_size)),
        )
        self.state_residual_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.state_residual_rr_gate = nn.Parameter(torch.tensor(-3.5))
        nn.init.zeros_(self.state_residual_hr_out[-1].weight)
        nn.init.zeros_(self.state_residual_hr_out[-1].bias)
        nn.init.zeros_(self.state_residual_rr_out[-1].weight)
        nn.init.zeros_(self.state_residual_rr_out[-1].bias)

    def forward(self, x):
        base = super().forward(x)
        hr_state, _ = self.state_residual_hr_gru(self.state_residual_hr_norm(self.last_hr_seq))
        rr_state, _ = self.state_residual_rr_gru(self.state_residual_rr_norm(self.last_rr_seq))
        self.last_hr_seq = self.last_hr_seq + torch.sigmoid(self.state_residual_hr_gate) * self.state_residual_hr_out(hr_state)
        self.last_rr_seq = self.last_rr_seq + torch.sigmoid(self.state_residual_rr_gate) * self.state_residual_rr_out(rr_state)
        return base


class SubharmonicTemporalFilterResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        self.temporal_filter_hr_norm = nn.LayerNorm(int(output_size))
        self.temporal_filter_rr_norm = nn.LayerNorm(int(output_size))
        self.temporal_filter_hr = nn.ModuleList(
            [
                nn.Conv1d(int(output_size), int(output_size), kernel_size=kernel, padding=(kernel // 2) * dilation, dilation=dilation, groups=int(output_size), bias=False)
                for kernel, dilation in ((5, 1), (9, 2), (15, 3))
            ]
        )
        self.temporal_filter_rr = nn.ModuleList(
            [
                nn.Conv1d(int(output_size), int(output_size), kernel_size=kernel, padding=(kernel // 2) * dilation, dilation=dilation, groups=int(output_size), bias=False)
                for kernel, dilation in ((15, 1), (31, 2), (45, 3))
            ]
        )
        self.temporal_filter_hr_mix = nn.Sequential(
            nn.Conv1d(int(output_size) * 3, int(output_size), kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )
        self.temporal_filter_rr_mix = nn.Sequential(
            nn.Conv1d(int(output_size) * 3, int(output_size), kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )
        self.temporal_filter_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.temporal_filter_rr_gate = nn.Parameter(torch.tensor(-3.0))
        for module in [*self.temporal_filter_hr, *self.temporal_filter_rr, self.temporal_filter_hr_mix[0], self.temporal_filter_rr_mix[0]]:
            nn.init.zeros_(module.weight)

    @staticmethod
    def _filtered(seq, norm, filters, mix):
        x = norm(seq).permute(0, 2, 1)
        return mix(torch.cat([layer(x) for layer in filters], dim=1)).permute(0, 2, 1)

    def forward(self, x):
        base = super().forward(x)
        hr_delta = self._filtered(self.last_hr_seq, self.temporal_filter_hr_norm, self.temporal_filter_hr, self.temporal_filter_hr_mix)
        rr_delta = self._filtered(self.last_rr_seq, self.temporal_filter_rr_norm, self.temporal_filter_rr, self.temporal_filter_rr_mix)
        self.last_hr_seq = self.last_hr_seq + torch.sigmoid(self.temporal_filter_hr_gate) * hr_delta
        self.last_rr_seq = self.last_rr_seq + torch.sigmoid(self.temporal_filter_rr_gate) * rr_delta
        return base


class SubharmonicDualPhaseResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        self.dual_phase_hr_delta = self._make_phase_head(output_size, hidden_dim, dropout)
        self.dual_phase_rr_delta = self._make_phase_head(output_size, hidden_dim, dropout)
        self.dual_phase_hr_amp = self._make_phase_head(output_size, hidden_dim, dropout)
        self.dual_phase_rr_amp = self._make_phase_head(output_size, hidden_dim, dropout)
        self.dual_phase_hr_out = self._make_phase_out(output_size, hidden_dim, dropout)
        self.dual_phase_rr_out = self._make_phase_out(output_size, hidden_dim, dropout)
        self.dual_phase_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.dual_phase_rr_gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.dual_phase_hr_out[-1].weight)
        nn.init.zeros_(self.dual_phase_hr_out[-1].bias)
        nn.init.zeros_(self.dual_phase_rr_out[-1].weight)
        nn.init.zeros_(self.dual_phase_rr_out[-1].bias)

    @staticmethod
    def _make_phase_head(output_size, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_phase_out(output_size, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(int(output_size) + 3),
            nn.Linear(int(output_size) + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(output_size)),
        )

    def _bounded_delta(self, logits, low_bpm, high_bpm):
        low = float(low_bpm) / (60.0 * float(self.fs))
        high = float(high_bpm) / (60.0 * float(self.fs))
        return low + torch.sigmoid(logits).squeeze(-1) * (high - low)

    def _phase_residual(self, seq, delta_head, amp_head, out_head, low_bpm, high_bpm):
        delta = self._bounded_delta(delta_head(seq), low_bpm, high_bpm)
        phase = torch.cumsum(delta, dim=1)
        angle = 2.0 * torch.pi * phase
        amp = torch.sigmoid(amp_head(seq)).squeeze(-1)
        phase_features = torch.stack([torch.sin(angle), torch.cos(angle), amp], dim=-1)
        return out_head(torch.cat([seq, phase_features], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_delta = self._phase_residual(
            self.last_hr_seq,
            self.dual_phase_hr_delta,
            self.dual_phase_hr_amp,
            self.dual_phase_hr_out,
            self.hr_low_bpm,
            self.hr_high_bpm,
        )
        rr_delta = self._phase_residual(
            self.last_rr_seq,
            self.dual_phase_rr_delta,
            self.dual_phase_rr_amp,
            self.dual_phase_rr_out,
            self.rr_low_bpm,
            self.rr_high_bpm,
        )
        self.last_hr_seq = self.last_hr_seq + torch.sigmoid(self.dual_phase_hr_gate) * hr_delta
        self.last_rr_seq = self.last_rr_seq + torch.sigmoid(self.dual_phase_rr_gate) * rr_delta
        return base


class SubharmonicParallelStateFusionMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_size = args[0] if len(args) > 0 else kwargs["input_size"]
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        groups = 8 if int(output_size) % 8 == 0 else 1
        self.parallel_state_input_norm = nn.LayerNorm(int(input_size))
        self.parallel_state_input_proj = nn.Linear(int(input_size), int(output_size))
        self.parallel_state_hr = nn.Sequential(
            nn.Conv1d(int(output_size), int(output_size), kernel_size=9, padding=4, groups=int(output_size), bias=False),
            nn.GroupNorm(groups, int(output_size)),
            nn.GELU(),
            nn.Conv1d(int(output_size), int(output_size), kernel_size=1, bias=False),
            nn.GroupNorm(groups, int(output_size)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.parallel_state_rr_pre = nn.Sequential(
            nn.Conv1d(int(output_size), int(output_size), kernel_size=31, padding=15, groups=int(output_size), bias=False),
            nn.GroupNorm(groups, int(output_size)),
            nn.GELU(),
        )
        self.parallel_state_rr_gru = nn.GRU(
            int(output_size), max(16, int(output_size) // 2), num_layers=1, batch_first=True, bidirectional=True
        )
        rr_hidden = max(16, int(output_size) // 2) * 2
        self.parallel_state_rr_out = nn.Sequential(
            nn.LayerNorm(rr_hidden),
            nn.Dropout(dropout),
            nn.Linear(rr_hidden, int(output_size)),
        )
        self.parallel_state_hr_out = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), int(output_size)),
        )
        self.parallel_state_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.parallel_state_rr_gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.parallel_state_hr_out[-1].weight)
        nn.init.zeros_(self.parallel_state_hr_out[-1].bias)
        nn.init.zeros_(self.parallel_state_rr_out[-1].weight)
        nn.init.zeros_(self.parallel_state_rr_out[-1].bias)

    def forward(self, x):
        base = super().forward(x)
        state = self.parallel_state_input_proj(self.parallel_state_input_norm(x)).permute(0, 2, 1)
        hr_state = self.parallel_state_hr(state).permute(0, 2, 1)
        rr_state = self.parallel_state_rr_pre(state).permute(0, 2, 1)
        rr_state, _ = self.parallel_state_rr_gru(rr_state)
        self.last_hr_seq = self.last_hr_seq + torch.sigmoid(self.parallel_state_hr_gate) * self.parallel_state_hr_out(hr_state)
        self.last_rr_seq = self.last_rr_seq + torch.sigmoid(self.parallel_state_rr_gate) * self.parallel_state_rr_out(rr_state)
        return base


class SubharmonicRRPhaseResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        self.rr_phase_delta = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rr_phase_amp = nn.Sequential(
            nn.LayerNorm(int(output_size)),
            nn.Linear(int(output_size), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rr_phase_out = nn.Sequential(
            nn.LayerNorm(int(output_size) + 3),
            nn.Linear(int(output_size) + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(output_size)),
        )
        self.rr_phase_gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.rr_phase_out[-1].weight)
        nn.init.zeros_(self.rr_phase_out[-1].bias)

    def _bounded_rr_delta(self, logits):
        low = float(self.rr_low_bpm) / (60.0 * float(self.fs))
        high = float(self.rr_high_bpm) / (60.0 * float(self.fs))
        return low + torch.sigmoid(logits).squeeze(-1) * (high - low)

    def forward(self, x):
        base = super().forward(x)
        rr_delta = self._bounded_rr_delta(self.rr_phase_delta(self.last_rr_seq))
        rr_phase = torch.cumsum(rr_delta, dim=1)
        rr_angle = 2.0 * torch.pi * rr_phase
        rr_amp = torch.sigmoid(self.rr_phase_amp(self.last_rr_seq)).squeeze(-1)
        rr_phase_features = torch.stack([torch.sin(rr_angle), torch.cos(rr_angle), rr_amp], dim=-1)
        rr_residual = self.rr_phase_out(torch.cat([self.last_rr_seq, rr_phase_features], dim=-1))
        self.last_rr_seq = self.last_rr_seq + torch.sigmoid(self.rr_phase_gate) * rr_residual
        return base


class SubharmonicSlowRRReadoutMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.slow_factor = 4
        self.slow_norm = nn.LayerNorm(hidden_dim)
        self.rr_slow_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.hr_env_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rr_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rr_slow_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rr_slow_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.rr_slow_gate = nn.Parameter(torch.tensor(-3.0))
        self.hr_env_gate = nn.Parameter(torch.tensor(-3.0))

    def _slow_encode(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.slow_factor,
            stride=self.slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.slow_norm(slow))
        slow_ctx, _ = self.rr_slow_self(self.slow_norm(slow), self.slow_norm(slow), self.slow_norm(slow), need_weights=False)
        return self.slow_norm(slow + slow_ctx)

    def _hr_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        return self.hr_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = SubharmonicCardioRespMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._slow_encode(rr_seq, self.rr_slow_gru)
        hr_env_slow = self._slow_encode(self._hr_envelope(hr_seq), self.hr_env_gru)
        rr_slow_delta, _ = self.rr_slow_decode(self.slow_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        rr_env_delta, _ = self.hr_env_decode(self.slow_norm(rr_seq), hr_env_slow, hr_env_slow, need_weights=False)
        rr_seq = rr_seq + torch.sigmoid(self.rr_slow_gate) * self.rr_slow_ffn(rr_slow_delta)
        rr_seq = rr_seq + torch.sigmoid(self.hr_env_gate) * self.rr_slow_ffn(rr_env_delta)
        self.last_rr_seq = rr_seq
        return base


class TriPriorCardioRespMemoryMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = int(args[1] if len(args) > 1 else kwargs["output_size"])
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if output_size % 4 == 0 else 1
        self.tri_slow_factor = 4
        self.tri_norm = nn.LayerNorm(output_size)
        self.tri_hr_env = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.tri_rr_slow_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tri_env_slow_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tri_state_gru = nn.GRU(output_size * 2, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tri_slow_self = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.tri_slow_to_rr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.tri_env_to_rr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.tri_state_to_rr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.tri_state_to_hr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.tri_rr_update = nn.Sequential(
            nn.LayerNorm(output_size * 4),
            nn.Linear(output_size * 4, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
        )
        self.tri_hr_update = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.tri_rr_gate = nn.Parameter(torch.tensor(-3.5))
        self.tri_hr_gate = nn.Parameter(torch.tensor(-4.5))
        nn.init.zeros_(self.tri_rr_update[-1].weight)
        nn.init.zeros_(self.tri_rr_update[-1].bias)
        nn.init.zeros_(self.tri_hr_update[-1].weight)
        nn.init.zeros_(self.tri_hr_update[-1].bias)

    def _tri_slow(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.tri_slow_factor,
            stride=self.tri_slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.tri_norm(slow))
        slow_ctx, _ = self.tri_slow_self(self.tri_norm(slow), self.tri_norm(slow), self.tri_norm(slow), need_weights=False)
        return self.tri_norm(slow + slow_ctx)

    def _tri_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        return self.tri_hr_env(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_env = self._tri_envelope(hr_seq)
        rr_slow = self._tri_slow(rr_seq, self.tri_rr_slow_gru)
        env_slow = self._tri_slow(hr_env, self.tri_env_slow_gru)
        state_seq, _ = self.tri_state_gru(torch.cat([self.tri_norm(rr_seq), self.tri_norm(hr_env)], dim=-1))
        rr_slow_ctx, _ = self.tri_slow_to_rr(self.tri_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        rr_env_ctx, _ = self.tri_env_to_rr(self.tri_norm(rr_seq), env_slow, env_slow, need_weights=False)
        rr_state_ctx, _ = self.tri_state_to_rr(self.tri_norm(rr_seq), self.tri_norm(state_seq), self.tri_norm(state_seq), need_weights=False)
        hr_state_ctx, _ = self.tri_state_to_hr(self.tri_norm(hr_seq), self.tri_norm(state_seq), self.tri_norm(state_seq), need_weights=False)
        rr_delta = self.tri_rr_update(torch.cat([rr_seq, rr_slow_ctx, rr_env_ctx, rr_state_ctx], dim=-1))
        hr_delta = self.tri_hr_update(torch.cat([hr_seq, hr_state_ctx], dim=-1))
        self.last_rr_seq = rr_seq + torch.sigmoid(self.tri_rr_gate) * rr_delta
        self.last_hr_seq = hr_seq + torch.sigmoid(self.tri_hr_gate) * hr_delta
        return base


class HarmonicFrequencyOperatorMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if int(output_size) % 4 == 0 else 1
        self.freq_operator_norm = nn.LayerNorm(output_size)
        self.hr_freq_operator_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_freq_operator_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_freq_operator_ffn = self._make_frequency_ffn(output_size, dropout)
        self.rr_freq_operator_ffn = self._make_frequency_ffn(output_size, dropout)
        self.hr_sin_proj = nn.Linear(output_size, output_size, bias=False)
        self.hr_cos_proj = nn.Linear(output_size, output_size, bias=False)
        self.rr_sin_proj = nn.Linear(output_size, output_size, bias=False)
        self.rr_cos_proj = nn.Linear(output_size, output_size, bias=False)
        self.hr_operator_gate = nn.Parameter(torch.tensor(-3.0))
        self.rr_operator_gate = nn.Parameter(torch.tensor(-3.0))
        self.operator_out_gate = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def _make_frequency_ffn(channels, dropout):
        return nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def _frequency_operator_delta(self, seq, freqs_hz, embed, attn, ffn, sin_proj, cos_proj):
        tokens = self._prototype_tokens(seq, freqs_hz, embed)
        token_delta, _ = attn(
            self.freq_operator_norm(tokens),
            self.freq_operator_norm(seq),
            self.freq_operator_norm(seq),
            need_weights=False,
        )
        tokens = tokens + ffn(token_delta)
        time = torch.arange(seq.shape[1], device=seq.device, dtype=torch.float32) / float(self.fs)
        phase = 2.0 * math.pi * time[:, None] * freqs_hz.to(device=seq.device, dtype=torch.float32)[None, :]
        sin_basis = torch.sin(phase).to(dtype=seq.dtype)
        cos_basis = torch.cos(phase).to(dtype=seq.dtype)
        sin_values = sin_proj(tokens)
        cos_values = cos_proj(tokens)
        scale = max(float(freqs_hz.numel()), 1.0) ** 0.5
        return (
            torch.einsum("tp,bph->bth", sin_basis, sin_values)
            + torch.einsum("tp,bph->bth", cos_basis, cos_values)
        ) / scale

    def forward(self, x):
        base = SubharmonicCardioRespMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_delta = self._frequency_operator_delta(
            hr_seq,
            self.hr_proto_freqs_hz,
            self.hr_proto_embed,
            self.hr_freq_operator_attn,
            self.hr_freq_operator_ffn,
            self.hr_sin_proj,
            self.hr_cos_proj,
        )
        rr_delta = self._frequency_operator_delta(
            rr_seq,
            self.rr_proto_freqs_hz,
            self.rr_proto_embed,
            self.rr_freq_operator_attn,
            self.rr_freq_operator_ffn,
            self.rr_sin_proj,
            self.rr_cos_proj,
        )
        self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_operator_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.rr_operator_gate) * rr_delta
        operator_out = self.fuse(torch.cat([self.last_hr_seq, self.last_rr_seq], dim=-1))
        return base + torch.sigmoid(self.operator_out_gate) * operator_out


class FactorizedPhysioSubspaceMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        factor_dim = max(32, int(output_size) // 4)
        groups = 8 if factor_dim % 8 == 0 else 1
        heads = 4 if factor_dim % 4 == 0 else 1
        self.factor_norm = nn.LayerNorm(output_size)
        self.shared_factor = nn.Linear(output_size, factor_dim)
        self.hr_factor = nn.Linear(output_size, factor_dim)
        self.rr_factor = nn.Linear(output_size, factor_dim)
        self.factor_shared_blocks = self._make_factor_blocks(factor_dim, groups, ((9, 1), (15, 2)), dropout)
        self.hr_factor_blocks = self._make_factor_blocks(factor_dim, groups, ((5, 1), (7, 2), (9, 4)), dropout)
        self.rr_factor_blocks = self._make_factor_blocks(factor_dim, groups, ((21, 1), (31, 2), (45, 3)), dropout)
        self.shared_tokens = nn.Parameter(torch.zeros(1, 4, factor_dim))
        self.hr_tokens = nn.Parameter(torch.zeros(1, 4, factor_dim))
        self.rr_tokens = nn.Parameter(torch.zeros(1, 4, factor_dim))
        nn.init.trunc_normal_(self.shared_tokens, std=0.02)
        nn.init.trunc_normal_(self.hr_tokens, std=0.02)
        nn.init.trunc_normal_(self.rr_tokens, std=0.02)
        self.shared_token_attn = nn.MultiheadAttention(factor_dim, heads, dropout=dropout, batch_first=True)
        self.hr_token_attn = nn.MultiheadAttention(factor_dim, heads, dropout=dropout, batch_first=True)
        self.rr_token_attn = nn.MultiheadAttention(factor_dim, heads, dropout=dropout, batch_first=True)
        self.hr_decode_attn = nn.MultiheadAttention(factor_dim, heads, dropout=dropout, batch_first=True)
        self.rr_decode_attn = nn.MultiheadAttention(factor_dim, heads, dropout=dropout, batch_first=True)
        self.factor_ffn = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, factor_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_dim * 2, factor_dim),
        )
        self.hr_decode = nn.Sequential(
            nn.LayerNorm(factor_dim * 2),
            nn.Linear(factor_dim * 2, output_size),
            nn.Dropout(dropout),
        )
        self.rr_decode = nn.Sequential(
            nn.LayerNorm(factor_dim * 2),
            nn.Linear(factor_dim * 2, output_size),
            nn.Dropout(dropout),
        )
        self.factor_out = nn.Linear(output_size * 2, output_size)
        self.hr_factor_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_factor_gate = nn.Parameter(torch.tensor(-2.0))
        self.factor_out_gate = nn.Parameter(torch.tensor(-3.5))
        self.factor_orth_loss = None
        self.factor_recon_loss = None

    @staticmethod
    def _make_factor_blocks(channels, groups, specs, dropout):
        blocks = []
        for kernel_size, dilation in specs:
            padding = (kernel_size // 2) * dilation
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
                    nn.GroupNorm(groups, channels),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_factor_blocks(x, blocks):
        y = x.permute(0, 2, 1)
        for block in blocks:
            y = y + block(y)
        return y.permute(0, 2, 1)

    @staticmethod
    def _remove_projection(task, shared):
        numerator = (task * shared).sum(dim=-1, keepdim=True)
        denominator = shared.square().sum(dim=-1, keepdim=True) + 1e-6
        return task - (numerator / denominator) * shared

    @staticmethod
    def _orthogonal_loss(shared, hr, rr):
        shared = F.normalize(shared, dim=-1)
        hr = F.normalize(hr, dim=-1)
        rr = F.normalize(rr, dim=-1)
        return (
            (shared * hr).sum(dim=-1).square().mean()
            + (shared * rr).sum(dim=-1).square().mean()
            + 0.5 * (hr * rr).sum(dim=-1).square().mean()
        )

    def _token_refine(self, seq, tokens, attn):
        batch_size = seq.shape[0]
        query = tokens.to(device=seq.device, dtype=seq.dtype).expand(batch_size, -1, -1)
        memory, _ = attn(query, seq, seq, need_weights=False)
        return query + self.factor_ffn(memory)

    def factor_regularization_loss(self):
        if self.factor_orth_loss is None or self.factor_recon_loss is None:
            return None
        return self.factor_orth_loss + 0.05 * self.factor_recon_loss

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        shared = self.shared_factor(self.factor_norm(0.5 * (hr_seq + rr_seq)))
        hr_factor = self.hr_factor(self.factor_norm(hr_seq))
        rr_factor = self.rr_factor(self.factor_norm(rr_seq))
        shared = self._run_factor_blocks(shared, self.factor_shared_blocks)
        hr_factor = self._run_factor_blocks(hr_factor, self.hr_factor_blocks)
        rr_factor = self._run_factor_blocks(rr_factor, self.rr_factor_blocks)
        hr_factor = self._remove_projection(hr_factor, shared)
        rr_factor = self._remove_projection(rr_factor, shared)
        shared_memory = self._token_refine(shared, self.shared_tokens, self.shared_token_attn)
        hr_memory = self._token_refine(hr_factor, self.hr_tokens, self.hr_token_attn)
        rr_memory = self._token_refine(rr_factor, self.rr_tokens, self.rr_token_attn)
        hr_ctx, _ = self.hr_decode_attn(hr_factor, torch.cat([shared_memory, hr_memory], dim=1), torch.cat([shared_memory, hr_memory], dim=1), need_weights=False)
        rr_ctx, _ = self.rr_decode_attn(rr_factor, torch.cat([shared_memory, rr_memory], dim=1), torch.cat([shared_memory, rr_memory], dim=1), need_weights=False)
        hr_delta = self.hr_decode(torch.cat([shared, hr_ctx], dim=-1))
        rr_delta = self.rr_decode(torch.cat([shared, rr_ctx], dim=-1))
        self.factor_orth_loss = self._orthogonal_loss(shared, hr_factor, rr_factor)
        self.factor_recon_loss = (
            F.mse_loss(self.hr_factor(self.factor_norm(hr_seq + torch.sigmoid(self.hr_factor_gate) * hr_delta)), hr_factor.detach())
            + F.mse_loss(self.rr_factor(self.factor_norm(rr_seq + torch.sigmoid(self.rr_factor_gate) * rr_delta)), rr_factor.detach())
        )
        self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_factor_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.rr_factor_gate) * rr_delta
        out = self.factor_out(torch.cat([self.last_hr_seq, self.last_rr_seq], dim=-1))
        return base + torch.sigmoid(self.factor_out_gate) * out


class StyleAdaptiveFactorizedPhysioSubspaceMixer(FactorizedPhysioSubspaceMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_size = args[0] if len(args) > 0 else kwargs["input_size"]
        factor_dim = self.shared_factor.out_features
        self.style_mlp = nn.Sequential(
            nn.LayerNorm(int(input_size) * 2),
            nn.Linear(int(input_size) * 2, factor_dim * 2),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(factor_dim * 2, factor_dim * 6),
        )
        self.style_gate = nn.Parameter(torch.tensor(-2.5))

    @staticmethod
    def _apply_style(stream, gamma, beta, gate):
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return stream + torch.sigmoid(gate) * (stream * gamma + beta)

    def forward(self, x):
        base = ComplexPhysioPrototypeMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        style = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
        params = self.style_mlp(style)
        shared_gamma, shared_beta, hr_gamma, hr_beta, rr_gamma, rr_beta = params.chunk(6, dim=-1)
        shared = self.shared_factor(self.factor_norm(0.5 * (hr_seq + rr_seq)))
        hr_factor = self.hr_factor(self.factor_norm(hr_seq))
        rr_factor = self.rr_factor(self.factor_norm(rr_seq))
        shared = self._run_factor_blocks(shared, self.factor_shared_blocks)
        hr_factor = self._run_factor_blocks(hr_factor, self.hr_factor_blocks)
        rr_factor = self._run_factor_blocks(rr_factor, self.rr_factor_blocks)
        shared = self._apply_style(shared, shared_gamma, shared_beta, self.style_gate)
        hr_factor = self._apply_style(hr_factor, hr_gamma, hr_beta, self.style_gate)
        rr_factor = self._apply_style(rr_factor, rr_gamma, rr_beta, self.style_gate)
        hr_factor = self._remove_projection(hr_factor, shared)
        rr_factor = self._remove_projection(rr_factor, shared)
        shared_memory = self._token_refine(shared, self.shared_tokens, self.shared_token_attn)
        hr_memory = self._token_refine(hr_factor, self.hr_tokens, self.hr_token_attn)
        rr_memory = self._token_refine(rr_factor, self.rr_tokens, self.rr_token_attn)
        hr_kv = torch.cat([shared_memory, hr_memory], dim=1)
        rr_kv = torch.cat([shared_memory, rr_memory], dim=1)
        hr_ctx, _ = self.hr_decode_attn(hr_factor, hr_kv, hr_kv, need_weights=False)
        rr_ctx, _ = self.rr_decode_attn(rr_factor, rr_kv, rr_kv, need_weights=False)
        hr_delta = self.hr_decode(torch.cat([shared, hr_ctx], dim=-1))
        rr_delta = self.rr_decode(torch.cat([shared, rr_ctx], dim=-1))
        self.factor_orth_loss = self._orthogonal_loss(shared, hr_factor, rr_factor)
        self.factor_recon_loss = (
            F.mse_loss(self.hr_factor(self.factor_norm(hr_seq + torch.sigmoid(self.hr_factor_gate) * hr_delta)), hr_factor.detach())
            + F.mse_loss(self.rr_factor(self.factor_norm(rr_seq + torch.sigmoid(self.rr_factor_gate) * rr_delta)), rr_factor.detach())
        )
        self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_factor_gate) * hr_delta
        self.last_rr_seq = rr_seq + torch.sigmoid(self.rr_factor_gate) * rr_delta
        out = self.factor_out(torch.cat([self.last_hr_seq, self.last_rr_seq], dim=-1))
        return base + torch.sigmoid(self.factor_out_gate) * out


class StateSpacePhysioMemoryMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if int(output_size) % 4 == 0 else 1
        self.state_norm = nn.LayerNorm(output_size)
        self.hr_state_proj = nn.Linear(output_size, output_size)
        self.rr_state_proj = nn.Linear(output_size, output_size)
        self.hr_state_tokens = nn.Parameter(torch.zeros(1, 8, output_size))
        self.rr_state_tokens = nn.Parameter(torch.zeros(1, 8, output_size))
        nn.init.trunc_normal_(self.hr_state_tokens, std=0.02)
        nn.init.trunc_normal_(self.rr_state_tokens, std=0.02)
        self.hr_state_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_state_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_rr_state_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_hr_state_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_decode_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_decode_attn = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_state_ffn = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
        )
        self.rr_state_ffn = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
        )
        self.hr_delta = nn.Sequential(nn.LayerNorm(output_size), nn.Linear(output_size, output_size), nn.Dropout(dropout))
        self.rr_delta = nn.Sequential(nn.LayerNorm(output_size), nn.Linear(output_size, output_size), nn.Dropout(dropout))
        self.state_out = nn.Linear(output_size * 2, output_size)
        self.hr_state_gate = nn.Parameter(torch.tensor(-2.5))
        self.rr_state_gate = nn.Parameter(torch.tensor(-2.5))
        self.state_cross_gate = nn.Parameter(torch.tensor(-2.0))
        self.state_out_gate = nn.Parameter(torch.tensor(-3.5))

    @staticmethod
    def _slow_sequence(seq):
        if seq.shape[1] < 4:
            return seq
        slow = F.avg_pool1d(seq.permute(0, 2, 1), kernel_size=4, stride=2, padding=1)
        return slow.permute(0, 2, 1)

    def _state_tokens(self, tokens, seq, attn, ffn):
        query = tokens.to(device=seq.device, dtype=seq.dtype).expand(seq.shape[0], -1, -1)
        memory, _ = attn(self.state_norm(query), self.state_norm(seq), self.state_norm(seq), need_weights=False)
        tokens = query + memory
        return tokens + ffn(tokens)

    def forward(self, x):
        base = ComplexPhysioPrototypeMemoryMixer.forward(self, x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_state = self.hr_state_proj(self.state_norm(hr_seq))
        rr_state = self.rr_state_proj(self.state_norm(rr_seq))
        rr_slow = self._slow_sequence(rr_state)

        hr_tokens = self._state_tokens(self.hr_state_tokens, hr_state, self.hr_state_attn, self.hr_state_ffn)
        rr_tokens = self._state_tokens(self.rr_state_tokens, rr_slow, self.rr_state_attn, self.rr_state_ffn)
        cross = torch.tanh(self.state_cross_gate)
        hr_cross, _ = self.hr_rr_state_attn(self.state_norm(hr_tokens), self.state_norm(rr_tokens), self.state_norm(rr_tokens), need_weights=False)
        rr_cross, _ = self.rr_hr_state_attn(self.state_norm(rr_tokens), self.state_norm(hr_tokens), self.state_norm(hr_tokens), need_weights=False)
        hr_tokens = hr_tokens + cross * hr_cross
        rr_tokens = rr_tokens + cross * rr_cross

        hr_ctx, _ = self.hr_decode_attn(self.state_norm(hr_state), self.state_norm(hr_tokens), self.state_norm(hr_tokens), need_weights=False)
        rr_ctx, _ = self.rr_decode_attn(self.state_norm(rr_state), self.state_norm(rr_tokens), self.state_norm(rr_tokens), need_weights=False)
        self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_state_gate) * self.hr_delta(hr_ctx)
        self.last_rr_seq = rr_seq + torch.sigmoid(self.rr_state_gate) * self.rr_delta(rr_ctx)
        out = self.state_out(torch.cat([self.last_hr_seq, self.last_rr_seq], dim=-1))
        return base + torch.sigmoid(self.state_out_gate) * out


class LowRankCardioRespResidualMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        rank = max(4, min(16, int(output_size) // 2))
        self.lowrank_hr_norm = nn.LayerNorm(output_size)
        self.lowrank_rr_norm = nn.LayerNorm(output_size)
        self.lowrank_hr_down = nn.Linear(output_size, rank, bias=False)
        self.lowrank_rr_down = nn.Linear(output_size, rank, bias=False)
        self.lowrank_rr_to_hr = nn.MultiheadAttention(rank, num_heads=1, dropout=kwargs.get("dropout", 0.1), batch_first=True)
        self.lowrank_hr_to_rr = nn.MultiheadAttention(rank, num_heads=1, dropout=kwargs.get("dropout", 0.1), batch_first=True)
        self.lowrank_hr_up = nn.Linear(rank, output_size, bias=False)
        self.lowrank_rr_up = nn.Linear(rank, output_size, bias=False)
        self.lowrank_fuse_norm = nn.LayerNorm(output_size * 2)
        self.lowrank_fuse_down = nn.Linear(output_size * 2, rank, bias=False)
        self.lowrank_fuse_up = nn.Linear(rank, output_size, bias=False)
        self.lowrank_hr_gate = nn.Parameter(torch.tensor(1.0))
        self.lowrank_rr_gate = nn.Parameter(torch.tensor(1.0))
        self.lowrank_output_gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.lowrank_hr_up.weight)
        nn.init.zeros_(self.lowrank_rr_up.weight)
        nn.init.zeros_(self.lowrank_fuse_up.weight)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_low = F.gelu(self.lowrank_hr_down(self.lowrank_hr_norm(hr_seq)))
        rr_low = F.gelu(self.lowrank_rr_down(self.lowrank_rr_norm(rr_seq)))
        hr_ctx, _ = self.lowrank_rr_to_hr(hr_low, rr_low, rr_low, need_weights=False)
        rr_ctx, _ = self.lowrank_hr_to_rr(rr_low, hr_low, hr_low, need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.lowrank_hr_gate) * self.lowrank_hr_up(hr_low + hr_ctx)
        rr_seq = rr_seq + torch.tanh(self.lowrank_rr_gate) * self.lowrank_rr_up(rr_low + rr_ctx)
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        fused = self.lowrank_fuse_norm(torch.cat([hr_seq, rr_seq], dim=-1))
        delta = self.lowrank_fuse_up(F.gelu(self.lowrank_fuse_down(fused)))
        return base + torch.tanh(self.lowrank_output_gate) * delta


class OscillatorSubharmonicHRResidualMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = args[1] if len(args) > 1 else kwargs["output_size"]
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        self.hr_phase_delta = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.hr_phase_amp = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.hr_phase_output = nn.Sequential(
            nn.LayerNorm(output_size + 3),
            nn.Linear(output_size + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )
        self.hr_phase_gate = nn.Parameter(torch.tensor(-4.0))

    def _bounded_hr_delta(self, logits):
        low = float(self.hr_low_bpm) / (60.0 * float(self.fs))
        high = float(self.hr_high_bpm) / (60.0 * float(self.fs))
        return low + torch.sigmoid(logits).squeeze(-1) * (high - low)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        if hr_seq is not None:
            hr_delta = self._bounded_hr_delta(self.hr_phase_delta(hr_seq))
            hr_phase = torch.cumsum(hr_delta, dim=1)
            hr_amp = torch.sigmoid(self.hr_phase_amp(hr_seq)).squeeze(-1)
            hr_angle = 2.0 * torch.pi * hr_phase
            phase_features = torch.stack([torch.sin(hr_angle), torch.cos(hr_angle), hr_amp], dim=-1)
            hr_residual = self.hr_phase_output(torch.cat([hr_seq, phase_features], dim=-1))
            self.last_hr_seq = hr_seq + torch.sigmoid(self.hr_phase_gate) * hr_residual
        return base


class PostTemporalLowRankAdapter(nn.Module):
    def __init__(self, feature_dim, rank=16, dropout=0.1):
        super().__init__()
        rank = max(4, min(int(rank), int(feature_dim)))
        self.shared_norm = nn.LayerNorm(feature_dim)
        self.hr_norm = nn.LayerNorm(feature_dim)
        self.rr_norm = nn.LayerNorm(feature_dim)
        self.shared_down = nn.Linear(feature_dim, rank, bias=False)
        self.hr_down = nn.Linear(feature_dim, rank, bias=False)
        self.rr_down = nn.Linear(feature_dim, rank, bias=False)
        self.hr_to_rr = nn.MultiheadAttention(rank, num_heads=1, dropout=dropout, batch_first=True)
        self.rr_to_hr = nn.MultiheadAttention(rank, num_heads=1, dropout=dropout, batch_first=True)
        self.shared_up = nn.Linear(rank, feature_dim, bias=False)
        self.hr_up = nn.Linear(rank, feature_dim, bias=False)
        self.rr_up = nn.Linear(rank, feature_dim, bias=False)
        self.shared_gate = nn.Parameter(torch.tensor(1.0))
        self.hr_gate = nn.Parameter(torch.tensor(1.0))
        self.rr_gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.shared_up.weight)
        nn.init.zeros_(self.hr_up.weight)
        nn.init.zeros_(self.rr_up.weight)

    def forward(self, z_seq, hr_seq, rr_seq):
        shared_low = F.gelu(self.shared_down(self.shared_norm(z_seq)))
        hr_low = F.gelu(self.hr_down(self.hr_norm(hr_seq)))
        rr_low = F.gelu(self.rr_down(self.rr_norm(rr_seq)))
        hr_ctx, _ = self.rr_to_hr(hr_low, rr_low, rr_low, need_weights=False)
        rr_ctx, _ = self.hr_to_rr(rr_low, hr_low, hr_low, need_weights=False)
        z_seq = z_seq + torch.tanh(self.shared_gate) * self.shared_up(shared_low)
        hr_seq = hr_seq + torch.tanh(self.hr_gate) * self.hr_up(hr_low + hr_ctx)
        rr_seq = rr_seq + torch.tanh(self.rr_gate) * self.rr_up(rr_low + rr_ctx)
        return z_seq, hr_seq, rr_seq


class HRAnchoredRRResidualAdapter(nn.Module):
    def __init__(self, feature_dim, rank=16, dropout=0.1):
        super().__init__()
        rank = max(4, min(int(rank), int(feature_dim)))
        self.hr_norm = nn.LayerNorm(feature_dim)
        self.rr_norm = nn.LayerNorm(feature_dim)
        self.hr_down = nn.Linear(feature_dim, rank, bias=False)
        self.rr_down = nn.Linear(feature_dim, rank, bias=False)
        self.hr_to_rr = nn.MultiheadAttention(rank, num_heads=1, dropout=dropout, batch_first=True)
        self.hr_up = nn.Linear(rank, feature_dim, bias=False)
        self.rr_up = nn.Linear(rank, feature_dim, bias=False)
        self.hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.rr_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.hr_up.weight)
        nn.init.zeros_(self.rr_up.weight)

    def forward(self, z_seq, hr_seq, rr_seq):
        hr_low = F.gelu(self.hr_down(self.hr_norm(hr_seq)))
        rr_low = F.gelu(self.rr_down(self.rr_norm(rr_seq)))
        rr_ctx, _ = self.hr_to_rr(rr_low, hr_low, hr_low, need_weights=False)
        hr_seq = hr_seq + torch.sigmoid(self.hr_gate) * self.hr_up(hr_low)
        rr_seq = rr_seq + torch.sigmoid(self.rr_gate) * self.rr_up(rr_low + rr_ctx)
        return z_seq, hr_seq, rr_seq


class EpisodicPhysioWhiteningAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.shared_norm = nn.LayerNorm(feature_dim)
        self.hr_norm = nn.LayerNorm(feature_dim)
        self.rr_norm = nn.LayerNorm(feature_dim)
        self.shared_film = self._make_film(feature_dim, hidden_dim, dropout)
        self.hr_film = self._make_film(feature_dim, hidden_dim, dropout)
        self.rr_film = self._make_film(feature_dim, hidden_dim, dropout)
        self.shared_residual = self._make_residual(feature_dim, hidden_dim, dropout)
        self.hr_residual = self._make_residual(feature_dim, hidden_dim, dropout)
        self.rr_residual = self._make_residual(feature_dim, hidden_dim, dropout)
        self.shared_gate = nn.Parameter(torch.tensor(-2.0))
        self.hr_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_gate = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _make_film(feature_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    @staticmethod
    def _make_residual(feature_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    @staticmethod
    def _adapt(seq, norm, film, residual, gate, eps=1e-5):
        seq_norm = norm(seq)
        mean = seq_norm.mean(dim=1, keepdim=True)
        std = seq_norm.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        whitened = (seq_norm - mean) / std
        style = torch.cat([mean.squeeze(1), torch.log(std.squeeze(1))], dim=-1)
        gamma_beta = film(style).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        corrected = whitened * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        corrected = corrected + residual(corrected)
        return seq + torch.sigmoid(gate) * corrected

    def forward(self, z_seq, hr_seq, rr_seq):
        z_seq = self._adapt(z_seq, self.shared_norm, self.shared_film, self.shared_residual, self.shared_gate)
        hr_seq = self._adapt(hr_seq, self.hr_norm, self.hr_film, self.hr_residual, self.hr_gate)
        rr_seq = self._adapt(rr_seq, self.rr_norm, self.rr_film, self.rr_residual, self.rr_gate)
        return z_seq, hr_seq, rr_seq


class TaskConditionedPhysioNormMemory(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_tokens=4, dropout=0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_tokens = max(2, int(num_tokens))
        heads = 4 if self.feature_dim % 4 == 0 else 1
        self.shared_norm = nn.LayerNorm(feature_dim)
        self.hr_norm = nn.LayerNorm(feature_dim)
        self.rr_norm = nn.LayerNorm(feature_dim)
        self.memory_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, feature_dim))
        nn.init.trunc_normal_(self.memory_tokens, std=0.02)
        self.shared_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.hr_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.rr_attn = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.hr_to_shared = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.rr_to_shared = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.shared_film = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.hr_film = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.rr_film = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.shared_gate = nn.Parameter(torch.tensor(1.0))
        self.hr_gate = nn.Parameter(torch.tensor(1.0))
        self.rr_gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.shared_film[-1].weight)
        nn.init.zeros_(self.shared_film[-1].bias)
        nn.init.zeros_(self.hr_film[-1].weight)
        nn.init.zeros_(self.hr_film[-1].bias)
        nn.init.zeros_(self.rr_film[-1].weight)
        nn.init.zeros_(self.rr_film[-1].bias)

    def _apply_film(self, seq, ctx, film, gate):
        pooled = torch.mean(ctx, dim=1, keepdim=True).expand(-1, seq.shape[1], -1)
        scale_shift = film(torch.cat([seq, pooled], dim=-1))
        scale, shift = scale_shift.chunk(2, dim=-1)
        delta = torch.tanh(scale) * seq + shift
        return seq + torch.tanh(gate) * delta

    def forward(self, z_seq, hr_seq, rr_seq):
        tokens = self.memory_tokens.to(device=z_seq.device, dtype=z_seq.dtype).expand(z_seq.shape[0], -1, -1)
        z_norm = self.shared_norm(z_seq)
        hr_norm = self.hr_norm(hr_seq)
        rr_norm = self.rr_norm(rr_seq)
        shared_ctx, _ = self.shared_attn(z_norm, tokens, tokens, need_weights=False)
        hr_ctx, _ = self.hr_attn(hr_norm, tokens, tokens, need_weights=False)
        rr_ctx, _ = self.rr_attn(rr_norm, tokens, tokens, need_weights=False)
        hr_to_shared, _ = self.hr_to_shared(z_norm, hr_norm, hr_norm, need_weights=False)
        rr_to_shared, _ = self.rr_to_shared(z_norm, rr_norm, rr_norm, need_weights=False)
        z_seq = self._apply_film(z_seq, shared_ctx + 0.5 * (hr_to_shared + rr_to_shared), self.shared_film, self.shared_gate)
        hr_seq = self._apply_film(hr_seq, hr_ctx + shared_ctx, self.hr_film, self.hr_gate)
        rr_seq = self._apply_film(rr_seq, rr_ctx + shared_ctx, self.rr_film, self.rr_gate)
        return z_seq, hr_seq, rr_seq


class PrototypePreservingSlowRRMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.slow_factor = 4
        self.slow_norm = nn.LayerNorm(hidden_dim)
        self.rr_slow_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.hr_env_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rr_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rr_slow_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rr_slow_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.rr_slow_gate = nn.Parameter(torch.tensor(-3.0))
        self.hr_env_gate = nn.Parameter(torch.tensor(-3.0))
        self.output_preserve_gate = nn.Parameter(torch.tensor(-3.0))

    def _slow_encode(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.slow_factor,
            stride=self.slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.slow_norm(slow))
        slow_ctx, _ = self.rr_slow_self(self.slow_norm(slow), self.slow_norm(slow), self.slow_norm(slow), need_weights=False)
        return self.slow_norm(slow + slow_ctx)

    def _hr_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        return self.hr_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._slow_encode(rr_seq, self.rr_slow_gru)
        hr_env_slow = self._slow_encode(self._hr_envelope(hr_seq), self.hr_env_gru)
        rr_slow_delta, _ = self.rr_slow_decode(self.slow_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        rr_env_delta, _ = self.hr_env_decode(self.slow_norm(rr_seq), hr_env_slow, hr_env_slow, need_weights=False)
        rr_seq = rr_seq + torch.sigmoid(self.rr_slow_gate) * self.rr_slow_ffn(rr_slow_delta)
        rr_seq = rr_seq + torch.sigmoid(self.hr_env_gate) * self.rr_slow_ffn(rr_env_delta)
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.output_preserve_gate) * self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))


class SubharmonicSlowRRMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.slow_factor = 4
        self.slow_norm = nn.LayerNorm(hidden_dim)
        self.rr_slow_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.hr_env_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rr_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.rr_slow_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.hr_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rr_slow_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.rr_slow_gate = nn.Parameter(torch.tensor(-3.0))
        self.hr_env_gate = nn.Parameter(torch.tensor(-3.0))
        self.output_preserve_gate = nn.Parameter(torch.tensor(-3.0))

    def _slow_encode(self, seq, gru):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.slow_factor,
            stride=self.slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = gru(self.slow_norm(slow))
        slow_ctx, _ = self.rr_slow_self(self.slow_norm(slow), self.slow_norm(slow), self.slow_norm(slow), need_weights=False)
        return self.slow_norm(slow + slow_ctx)

    def _hr_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        return self.hr_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._slow_encode(rr_seq, self.rr_slow_gru)
        hr_env_slow = self._slow_encode(self._hr_envelope(hr_seq), self.hr_env_gru)
        rr_slow_delta, _ = self.rr_slow_decode(self.slow_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        rr_env_delta, _ = self.hr_env_decode(self.slow_norm(rr_seq), hr_env_slow, hr_env_slow, need_weights=False)
        rr_seq = rr_seq + torch.sigmoid(self.rr_slow_gate) * self.rr_slow_ffn(rr_slow_delta)
        rr_seq = rr_seq + torch.sigmoid(self.hr_env_gate) * self.rr_slow_ffn(rr_env_delta)
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.output_preserve_gate) * self.fuse(torch.cat([hr_seq, rr_seq], dim=-1))


class CardioRespEnvelopeMemoryMixer(ComplexPhysioPrototypeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.carrier_energy_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.envelope_blocks = nn.ModuleList()
        for kernel_size, dilation in ((15, 2), (31, 4), (51, 8)):
            padding = (kernel_size // 2) * dilation
            self.envelope_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.envelope_norm = nn.LayerNorm(hidden_dim)
        self.envelope_to_rr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.rr_to_carrier_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.carrier_modulation = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.envelope_gate = nn.Parameter(torch.tensor(0.0))
        self.carrier_gate = nn.Parameter(torch.tensor(0.0))

    def _carrier_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        energy = torch.cat([centered.abs(), centered.square()], dim=-1)
        envelope = self.carrier_energy_proj(energy)
        envelope_conv = envelope.permute(0, 2, 1)
        for block in self.envelope_blocks:
            envelope_conv = envelope_conv + block(envelope_conv)
        return self.envelope_norm(envelope_conv.permute(0, 2, 1))

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        envelope = self._carrier_envelope(hr_seq)
        rr_envelope, _ = self.envelope_to_rr_attn(self.norm(rr_seq), envelope, envelope, need_weights=False)
        carrier_context, _ = self.rr_to_carrier_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_seq = rr_seq + torch.tanh(self.envelope_gate) * rr_envelope
        hr_seq = hr_seq * (1.0 + torch.tanh(self.carrier_gate) * self.carrier_modulation(carrier_context))

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class CardioRespStateSpaceMemoryMixer(CardioRespEnvelopeMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_freq_embed.shape[-1]
        dropout = kwargs.get("dropout", 0.1)
        self.resp_state_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.resp_state_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.resp_state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )
        self.state_to_rr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.state_to_hr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.resp_state_gate = nn.Parameter(torch.tensor(0.0))
        self.hr_state_gate = nn.Parameter(torch.tensor(-2.0))

    def _resp_state_tokens(self, envelope):
        state_seq, _ = self.resp_state_gru(envelope)
        pooled = torch.cat(
            [
                torch.mean(state_seq, dim=1),
                torch.std(state_seq, dim=1, unbiased=False),
                torch.amax(state_seq, dim=1),
                torch.amin(state_seq, dim=1),
            ],
            dim=-1,
        )
        global_token = self.resp_state_proj(pooled).unsqueeze(1)
        tokens = torch.cat([state_seq, global_token], dim=1)
        return self.resp_state_encoder(tokens)

    def forward(self, x):
        x_norm = self.input_norm(x)
        shared = self.input_proj(x_norm)
        shared_conv = shared.permute(0, 2, 1)
        for block in self.shared_blocks:
            shared_conv = shared_conv + block(shared_conv)
        shared = shared_conv.permute(0, 2, 1)

        hr_conv = shared.permute(0, 2, 1)
        rr_conv = shared.permute(0, 2, 1)
        for hr_block, rr_block in zip(self.hr_blocks, self.rr_blocks):
            hr_conv = hr_conv + hr_block(hr_conv)
            rr_conv = rr_conv + rr_block(rr_conv)
        hr_seq = hr_conv.permute(0, 2, 1)
        rr_seq = rr_conv.permute(0, 2, 1)

        hr_tokens = self._band_tokens(hr_seq, self.hr_low_bpm, self.hr_high_bpm, self.hr_freq_embed)
        rr_tokens = self._band_tokens(rr_seq, self.rr_low_bpm, self.rr_high_bpm, self.rr_freq_embed)
        hr_band, _ = self.hr_band_attn(self.norm(hr_seq), self.norm(hr_tokens), self.norm(hr_tokens), need_weights=False)
        rr_band, _ = self.rr_band_attn(self.norm(rr_seq), self.norm(rr_tokens), self.norm(rr_tokens), need_weights=False)
        hr_seq = hr_seq + hr_band
        rr_seq = rr_seq + rr_band

        hr_proto = self._prototype_tokens(hr_seq, self.hr_proto_freqs_hz, self.hr_proto_embed)
        rr_proto = self._prototype_tokens(rr_seq, self.rr_proto_freqs_hz, self.rr_proto_embed)
        hr_memory, _ = self.hr_proto_attn(self.norm(hr_seq), self.norm(hr_proto), self.norm(hr_proto), need_weights=False)
        rr_memory, _ = self.rr_proto_attn(self.norm(rr_seq), self.norm(rr_proto), self.norm(rr_proto), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_proto_gate) * hr_memory
        rr_seq = rr_seq + torch.tanh(self.rr_proto_gate) * rr_memory

        envelope = self._carrier_envelope(hr_seq)
        resp_state = self._resp_state_tokens(envelope)
        rr_envelope, _ = self.envelope_to_rr_attn(self.norm(rr_seq), envelope, envelope, need_weights=False)
        rr_state, _ = self.state_to_rr_attn(self.norm(rr_seq), self.norm(resp_state), self.norm(resp_state), need_weights=False)
        carrier_context, _ = self.rr_to_carrier_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        hr_state, _ = self.state_to_hr_attn(self.norm(hr_seq), self.norm(resp_state), self.norm(resp_state), need_weights=False)
        rr_seq = rr_seq + torch.tanh(self.envelope_gate) * rr_envelope + torch.tanh(self.resp_state_gate) * rr_state
        hr_seq = hr_seq * (1.0 + torch.tanh(self.carrier_gate) * self.carrier_modulation(carrier_context))
        hr_seq = hr_seq + torch.tanh(self.hr_state_gate) * hr_state

        hr_cross, _ = self.hr_from_rr_attn(self.norm(hr_seq), self.norm(rr_seq), self.norm(rr_seq), need_weights=False)
        rr_cross, _ = self.rr_from_hr_attn(self.norm(rr_seq), self.norm(hr_seq), self.norm(hr_seq), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_cross_gate) * hr_cross
        rr_seq = rr_seq + torch.tanh(self.rr_cross_gate) * rr_cross
        hr_seq = hr_seq + self.hr_ffn(hr_seq)
        rr_seq = rr_seq + self.rr_ffn(rr_seq)

        self.last_hr_seq = self.hr_output(hr_seq)
        self.last_rr_seq = self.rr_output(rr_seq)
        weights = torch.softmax(self.fusion_gate(torch.cat([hr_seq, rr_seq], dim=-1)), dim=-1)
        mixed = torch.cat([hr_seq * weights[..., :1], rr_seq * weights[..., 1:]], dim=-1)
        return self.fuse(mixed) + self.skip_proj(x_norm)


class AdaptivePriorCardioRespMemoryMixer(SubharmonicCardioRespMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hr_output.out_features
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if hidden_dim % 4 == 0 else 1
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.adaptive_slow_factor = 4
        self.adaptive_norm = nn.LayerNorm(hidden_dim)
        self.adaptive_rr_slow_gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.adaptive_state_gru = nn.GRU(hidden_dim * 2, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.adaptive_slow_self = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.adaptive_slow_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.adaptive_state_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.adaptive_hr_env_decode = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.adaptive_env_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.adaptive_env_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=31, padding=15, groups=hidden_dim, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adaptive_rr_candidates = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(3)
            ]
        )
        self.adaptive_hr_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adaptive_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.adaptive_output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.adaptive_rr_gate = nn.Parameter(torch.tensor(-3.0))
        self.adaptive_hr_gate = nn.Parameter(torch.tensor(-4.0))
        self.adaptive_output_gate = nn.Parameter(torch.tensor(-4.0))

    def _adaptive_slow_tokens(self, seq):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.adaptive_slow_factor,
            stride=self.adaptive_slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = self.adaptive_rr_slow_gru(self.adaptive_norm(slow))
        slow_ctx, _ = self.adaptive_slow_self(self.adaptive_norm(slow), self.adaptive_norm(slow), self.adaptive_norm(slow), need_weights=False)
        return self.adaptive_norm(slow + slow_ctx)

    def _adaptive_envelope(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        env = self.adaptive_env_proj(torch.cat([centered.abs(), centered.square()], dim=-1))
        env_conv = self.adaptive_env_conv(env.permute(0, 2, 1)).permute(0, 2, 1)
        return self.adaptive_norm(env + env_conv)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._adaptive_slow_tokens(rr_seq)
        hr_env = self._adaptive_envelope(hr_seq)
        state_input = torch.cat([rr_seq, hr_env], dim=-1)
        state_seq, _ = self.adaptive_state_gru(state_input)
        rr_slow_ctx, _ = self.adaptive_slow_decode(self.adaptive_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        rr_state_ctx, _ = self.adaptive_state_decode(self.adaptive_norm(rr_seq), self.adaptive_norm(state_seq), self.adaptive_norm(state_seq), need_weights=False)
        rr_env_ctx, _ = self.adaptive_hr_env_decode(self.adaptive_norm(rr_seq), self.adaptive_norm(hr_env), self.adaptive_norm(hr_env), need_weights=False)
        candidates = torch.stack(
            [
                self.adaptive_rr_candidates[0](rr_slow_ctx),
                self.adaptive_rr_candidates[1](rr_state_ctx),
                self.adaptive_rr_candidates[2](rr_env_ctx),
            ],
            dim=2,
        )
        pooled = torch.cat(
            [
                torch.mean(hr_seq, dim=1),
                torch.std(hr_seq, dim=1, unbiased=False),
                torch.mean(rr_seq, dim=1),
                torch.std(rr_seq, dim=1, unbiased=False),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.adaptive_gate(pooled), dim=-1).view(hr_seq.shape[0], 1, 3, 1)
        self.last_adaptive_prior_weights = weights.detach().mean(dim=(0, 1, 3))
        self.last_adaptive_prior_gate_values = torch.stack(
            [
                torch.sigmoid(self.adaptive_rr_gate),
                torch.sigmoid(self.adaptive_hr_gate),
                torch.sigmoid(self.adaptive_output_gate),
            ]
        ).detach()
        rr_delta = torch.sum(candidates * weights, dim=2)
        hr_delta = self.adaptive_hr_residual(rr_env_ctx)
        rr_seq = rr_seq + torch.sigmoid(self.adaptive_rr_gate) * rr_delta
        hr_seq = hr_seq + torch.sigmoid(self.adaptive_hr_gate) * hr_delta
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        return base + torch.sigmoid(self.adaptive_output_gate) * self.adaptive_output_proj(torch.cat([hr_seq, rr_seq], dim=-1))


class ContinuousCardioRespStateMixer(CardioRespStateSpaceMemoryMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = int(kwargs.get("output_size", args[1] if len(args) > 1 else 128))
        dropout = kwargs.get("dropout", 0.1)
        groups = 8 if output_size % 8 == 0 else 1
        self.hr_state_norm = nn.LayerNorm(output_size)
        self.rr_state_norm = nn.LayerNorm(output_size)
        self.hr_rate_head = nn.Sequential(
            nn.LayerNorm(output_size * 4),
            nn.Linear(output_size * 4, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1),
        )
        self.rr_rate_head = nn.Sequential(
            nn.LayerNorm(output_size * 4),
            nn.Linear(output_size * 4, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, 1),
        )
        self.hr_amp_head = nn.Sequential(nn.LayerNorm(output_size), nn.Linear(output_size, 1), nn.Sigmoid())
        self.rr_amp_head = nn.Sequential(nn.LayerNorm(output_size), nn.Linear(output_size, 1), nn.Sigmoid())
        self.hr_phase_proj = nn.Sequential(nn.Linear(3, output_size), nn.GELU(), nn.Linear(output_size, output_size))
        self.rr_phase_proj = nn.Sequential(nn.Linear(3, output_size), nn.GELU(), nn.Linear(output_size, output_size))
        self.hr_state_filter = nn.Sequential(
            nn.Conv1d(output_size, output_size, kernel_size=7, padding=3, groups=output_size, bias=False),
            nn.GroupNorm(groups, output_size),
            nn.GELU(),
            nn.Conv1d(output_size, output_size, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )
        self.rr_state_filter = nn.Sequential(
            nn.Conv1d(output_size, output_size, kernel_size=21, padding=10, groups=output_size, bias=False),
            nn.GroupNorm(groups, output_size),
            nn.GELU(),
            nn.Conv1d(output_size, output_size, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )
        heads = 4 if output_size % 4 == 0 else 1
        self.hr_state_to_rr = nn.MultiheadAttention(output_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.rr_state_to_hr = nn.MultiheadAttention(output_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.state_fuse = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.hr_state_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_state_gate = nn.Parameter(torch.tensor(0.0))
        self.cross_state_gate = nn.Parameter(torch.tensor(0.0))
        self.output_state_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _summary(seq):
        return torch.cat(
            [
                torch.mean(seq, dim=1),
                torch.std(seq, dim=1, unbiased=False),
                torch.amax(seq, dim=1),
                torch.amin(seq, dim=1),
            ],
            dim=-1,
        )

    def _state_tokens(self, seq, norm, rate_head, amp_head, phase_proj, state_filter, low_bpm, high_bpm):
        seq_norm = norm(seq)
        rate_unit = torch.sigmoid(rate_head(self._summary(seq_norm))).squeeze(-1)
        rate_bpm = float(low_bpm) + rate_unit * (float(high_bpm) - float(low_bpm))
        time = torch.arange(seq.shape[1], device=seq.device, dtype=seq.dtype)[None, :] / float(self.fs)
        phase = 2.0 * math.pi * (rate_bpm[:, None].to(dtype=seq.dtype) / 60.0) * time
        amp = amp_head(seq_norm).squeeze(-1)
        phase_features = torch.stack([torch.sin(phase), torch.cos(phase), amp], dim=-1)
        state = phase_proj(phase_features)
        state = state + state_filter(state.permute(0, 2, 1)).permute(0, 2, 1)
        return state, rate_bpm

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_state, hr_state_rate_bpm = self._state_tokens(
            hr_seq,
            self.hr_state_norm,
            self.hr_rate_head,
            self.hr_amp_head,
            self.hr_phase_proj,
            self.hr_state_filter,
            self.hr_low_bpm,
            self.hr_high_bpm,
        )
        rr_state, rr_state_rate_bpm = self._state_tokens(
            rr_seq,
            self.rr_state_norm,
            self.rr_rate_head,
            self.rr_amp_head,
            self.rr_phase_proj,
            self.rr_state_filter,
            self.rr_low_bpm,
            self.rr_high_bpm,
        )
        self.last_state_rate_bpm = {"hr": hr_state_rate_bpm, "rr": rr_state_rate_bpm}
        hr_ctx, _ = self.rr_state_to_hr(self.hr_state_norm(hr_seq), self.rr_state_norm(rr_state), self.rr_state_norm(rr_state), need_weights=False)
        rr_ctx, _ = self.hr_state_to_rr(self.rr_state_norm(rr_seq), self.hr_state_norm(hr_state), self.hr_state_norm(hr_state), need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_state_gate) * hr_state + torch.tanh(self.cross_state_gate) * hr_ctx
        rr_seq = rr_seq + torch.tanh(self.rr_state_gate) * rr_state + torch.tanh(self.cross_state_gate) * rr_ctx
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        state_delta = self.state_fuse(torch.cat([hr_seq, rr_seq], dim=-1))
        return base + torch.tanh(self.output_state_gate) * state_delta


class AdaptiveCardioRespStateMixer(ContinuousCardioRespStateMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = int(kwargs.get("output_size", args[1] if len(args) > 1 else 128))
        dropout = kwargs.get("dropout", 0.1)
        self.hr_rate_gru = nn.GRU(
            input_size=output_size,
            hidden_size=output_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rr_rate_gru = nn.GRU(
            input_size=output_size,
            hidden_size=output_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.hr_rate_seq_head = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size // 2, 1),
        )
        self.rr_rate_seq_head = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size // 2, 1),
        )
        self.hr_rate_blend = nn.Parameter(torch.tensor(0.0))
        self.rr_rate_blend = nn.Parameter(torch.tensor(0.0))

    def _adaptive_state_tokens(
        self,
        seq,
        norm,
        rate_head,
        rate_gru,
        rate_seq_head,
        amp_head,
        phase_proj,
        state_filter,
        low_bpm,
        high_bpm,
        blend_gate,
    ):
        seq_norm = norm(seq)
        global_unit = torch.sigmoid(rate_head(self._summary(seq_norm))).squeeze(-1).unsqueeze(1)
        rate_features, _ = rate_gru(seq_norm)
        local_unit = torch.sigmoid(rate_seq_head(rate_features).squeeze(-1))
        blend = torch.sigmoid(blend_gate)
        rate_unit = blend * local_unit + (1.0 - blend) * global_unit
        rate_bpm_seq = float(low_bpm) + rate_unit * (float(high_bpm) - float(low_bpm))
        phase_step = 2.0 * math.pi * rate_bpm_seq.to(dtype=seq.dtype) / (60.0 * float(self.fs))
        phase = torch.cumsum(phase_step, dim=1)
        amp = amp_head(seq_norm).squeeze(-1)
        phase_features = torch.stack([torch.sin(phase), torch.cos(phase), amp], dim=-1)
        state = phase_proj(phase_features)
        state = state + state_filter(state.permute(0, 2, 1)).permute(0, 2, 1)
        return state, rate_bpm_seq.mean(dim=1)

    def _state_tokens(self, seq, norm, rate_head, amp_head, phase_proj, state_filter, low_bpm, high_bpm):
        if rate_head is self.hr_rate_head:
            return self._adaptive_state_tokens(
                seq,
                norm,
                rate_head,
                self.hr_rate_gru,
                self.hr_rate_seq_head,
                amp_head,
                phase_proj,
                state_filter,
                low_bpm,
                high_bpm,
                self.hr_rate_blend,
            )
        return self._adaptive_state_tokens(
            seq,
            norm,
            rate_head,
            self.rr_rate_gru,
            self.rr_rate_seq_head,
            amp_head,
            phase_proj,
            state_filter,
            low_bpm,
            high_bpm,
            self.rr_rate_blend,
        )


class LatentCardioRespDynamicsMixer(AdaptiveCardioRespStateMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = int(kwargs.get("output_size", args[1] if len(args) > 1 else 128))
        dropout = kwargs.get("dropout", 0.1)
        num_freq_bins = int(kwargs.get("num_freq_bins", 96))
        latent_tokens = min(max(num_freq_bins // 6, 8), 16)
        heads = 4 if output_size % 4 == 0 else 1
        self.hr_dynamics_query = nn.Parameter(torch.zeros(1, latent_tokens, output_size))
        self.rr_dynamics_query = nn.Parameter(torch.zeros(1, latent_tokens, output_size))
        nn.init.trunc_normal_(self.hr_dynamics_query, std=0.02)
        nn.init.trunc_normal_(self.rr_dynamics_query, std=0.02)
        self.dynamics_norm = nn.LayerNorm(output_size)
        self.hr_dynamics_encoder = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_dynamics_encoder = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_dynamics_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rr_dynamics_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.hr_to_rr_dynamics = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_to_hr_dynamics = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_dynamics_decoder = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_dynamics_decoder = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_dynamics_ffn = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
        )
        self.rr_dynamics_ffn = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
        )
        self.dynamics_fuse = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.hr_dynamics_gate = nn.Parameter(torch.tensor(0.0))
        self.rr_dynamics_gate = nn.Parameter(torch.tensor(0.0))
        self.cross_dynamics_gate = nn.Parameter(torch.tensor(0.0))
        self.output_dynamics_gate = nn.Parameter(torch.tensor(0.0))

    def _dynamics_tokens(self, seq, query, encoder, dynamics_gru):
        batch = seq.shape[0]
        tokens = query.to(device=seq.device, dtype=seq.dtype).expand(batch, -1, -1)
        seq_norm = self.dynamics_norm(seq)
        tokens, _ = encoder(tokens, seq_norm, seq_norm, need_weights=False)
        tokens, _ = dynamics_gru(tokens)
        return self.dynamics_norm(tokens)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        hr_latent = self._dynamics_tokens(
            hr_seq,
            self.hr_dynamics_query,
            self.hr_dynamics_encoder,
            self.hr_dynamics_gru,
        )
        rr_latent = self._dynamics_tokens(
            rr_seq,
            self.rr_dynamics_query,
            self.rr_dynamics_encoder,
            self.rr_dynamics_gru,
        )
        hr_cross, _ = self.rr_to_hr_dynamics(hr_latent, rr_latent, rr_latent, need_weights=False)
        rr_cross, _ = self.hr_to_rr_dynamics(rr_latent, hr_latent, hr_latent, need_weights=False)
        hr_latent = self.dynamics_norm(hr_latent + torch.tanh(self.cross_dynamics_gate) * hr_cross)
        rr_latent = self.dynamics_norm(rr_latent + torch.tanh(self.cross_dynamics_gate) * rr_cross)
        hr_delta, _ = self.hr_dynamics_decoder(self.dynamics_norm(hr_seq), hr_latent, hr_latent, need_weights=False)
        rr_delta, _ = self.rr_dynamics_decoder(self.dynamics_norm(rr_seq), rr_latent, rr_latent, need_weights=False)
        hr_seq = hr_seq + torch.tanh(self.hr_dynamics_gate) * self.hr_dynamics_ffn(hr_delta)
        rr_seq = rr_seq + torch.tanh(self.rr_dynamics_gate) * self.rr_dynamics_ffn(rr_delta)
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        dynamics_delta = self.dynamics_fuse(torch.cat([hr_seq, rr_seq], dim=-1))
        return base + torch.tanh(self.output_dynamics_gate) * dynamics_delta


class HierarchicalCardioRespDynamicsMixer(LatentCardioRespDynamicsMixer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_size = int(kwargs.get("output_size", args[1] if len(args) > 1 else 128))
        dropout = kwargs.get("dropout", 0.1)
        heads = 4 if output_size % 4 == 0 else 1
        self.rr_slow_factor = 4
        self.rr_slow_norm = nn.LayerNorm(output_size)
        self.rr_slow_gru = nn.GRU(output_size, output_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rr_slow_self = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_slow_to_rr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.rr_slow_to_hr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_envelope_to_rr = nn.MultiheadAttention(output_size, heads, dropout=dropout, batch_first=True)
        self.hr_envelope_proj = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.hier_fuse = nn.Sequential(
            nn.LayerNorm(output_size * 2),
            nn.Linear(output_size * 2, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size),
        )
        self.rr_slow_gate = nn.Parameter(torch.tensor(0.0))
        self.hr_slow_gate = nn.Parameter(torch.tensor(-1.0))
        self.hr_envelope_gate = nn.Parameter(torch.tensor(0.0))
        self.hier_output_gate = nn.Parameter(torch.tensor(0.0))

    def _slow_tokens(self, seq):
        slow = F.avg_pool1d(
            seq.permute(0, 2, 1),
            kernel_size=self.rr_slow_factor,
            stride=self.rr_slow_factor,
            ceil_mode=True,
        ).permute(0, 2, 1)
        slow, _ = self.rr_slow_gru(self.rr_slow_norm(slow))
        slow_ctx, _ = self.rr_slow_self(self.rr_slow_norm(slow), self.rr_slow_norm(slow), self.rr_slow_norm(slow), need_weights=False)
        return self.rr_slow_norm(slow + slow_ctx)

    def _hr_envelope_tokens(self, hr_seq):
        centered = hr_seq - torch.mean(hr_seq, dim=1, keepdim=True)
        envelope = self.hr_envelope_proj(torch.cat([centered.abs(), centered.square()], dim=-1))
        return self._slow_tokens(envelope)

    def forward(self, x):
        base = super().forward(x)
        hr_seq = self.last_hr_seq
        rr_seq = self.last_rr_seq
        rr_slow = self._slow_tokens(rr_seq)
        hr_env_slow = self._hr_envelope_tokens(hr_seq)
        rr_slow_ctx, _ = self.rr_slow_to_rr(self.rr_slow_norm(rr_seq), rr_slow, rr_slow, need_weights=False)
        hr_slow_ctx, _ = self.rr_slow_to_hr(self.rr_slow_norm(hr_seq), rr_slow, rr_slow, need_weights=False)
        rr_env_ctx, _ = self.hr_envelope_to_rr(self.rr_slow_norm(rr_seq), hr_env_slow, hr_env_slow, need_weights=False)
        rr_seq = rr_seq + torch.tanh(self.rr_slow_gate) * rr_slow_ctx + torch.tanh(self.hr_envelope_gate) * rr_env_ctx
        hr_seq = hr_seq + torch.tanh(self.hr_slow_gate) * hr_slow_ctx
        self.last_hr_seq = hr_seq
        self.last_rr_seq = rr_seq
        hier_delta = self.hier_fuse(torch.cat([hr_seq, rr_seq], dim=-1))
        return base + torch.tanh(self.hier_output_gate) * hier_delta


class BottleneckTemporalHRHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hr_num_bins, bottleneck_dim=32, dropout=0.1):
        super().__init__()
        self.token = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        groups = 8 if bottleneck_dim % 8 == 0 else 1
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (9, 1), (9, 2), (15, 2)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        bottleneck_dim,
                        bottleneck_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=bottleneck_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, bottleneck_dim),
                    nn.GELU(),
                    nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, bottleneck_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=4 if bottleneck_dim % 4 == 0 else 1,
            dim_feedforward=max(hidden_dim, bottleneck_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention = nn.Linear(bottleneck_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(bottleneck_dim * 5),
            nn.Linear(bottleneck_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hr_num_bins),
        )

    def forward(self, z_seq):
        x = self.token(z_seq)
        y = x.permute(0, 2, 1)
        for block in self.blocks:
            y = y + block(y)
        x = y.permute(0, 2, 1)
        x = self.context(x)
        weights = torch.softmax(self.attention(x), dim=1)
        weighted = torch.sum(x * weights, dim=1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=1),
                torch.std(x, dim=1, unbiased=False),
                torch.amax(x, dim=1),
                torch.amin(x, dim=1),
            ],
            dim=1,
        )
        return self.classifier(pooled)


class MultiScaleWaveformHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.in_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (5, 2), (7, 4), (7, 8), (9, 16)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.out = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, z_seq):
        x = self.in_proj(z_seq.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        return self.out(x).squeeze(1)


class FrequencyConditionedWaveformDecoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        hr_num_bins=128,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        fs=30.0,
        dropout=0.1,
    ):
        super().__init__()
        self.hr_num_bins = int(hr_num_bins)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.fs = float(fs)
        self.residual_head = MultiScaleWaveformHead(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        param_out = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(param_out.weight)
        nn.init.zeros_(param_out.bias)
        with torch.no_grad():
            param_out.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, -2.0]))
        self.param_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            param_out,
        )

    def forward(self, z_seq, hr_logits, frame_offsets=None):
        batch_size, frames, _ = z_seq.shape
        residual = self.residual_head(z_seq)
        pooled = torch.cat(
            [
                torch.mean(z_seq, dim=1),
                torch.std(z_seq, dim=1, unbiased=False),
                torch.amax(z_seq, dim=1),
                torch.amin(z_seq, dim=1),
            ],
            dim=1,
        )
        params = self.param_head(pooled)
        coeffs = torch.tanh(params[:, :4])
        gate = torch.sigmoid(params[:, 4:5])
        residual_gate = torch.sigmoid(params[:, 5:6])

        weights = torch.softmax(hr_logits, dim=1)
        bpm = torch.linspace(self.hr_low_bpm, self.hr_high_bpm, self.hr_num_bins, device=z_seq.device)
        hz = torch.sum(weights * (bpm / 60.0).unsqueeze(0), dim=1)
        frame_index = torch.arange(frames, device=z_seq.device, dtype=z_seq.dtype)
        if frame_offsets is not None:
            frame_offsets = frame_offsets.to(device=z_seq.device, dtype=z_seq.dtype)
            frame_index = frame_index[None, :] + frame_offsets[:, None]
        else:
            frame_index = frame_index[None, :]
        time = frame_index / self.fs
        angles = 2.0 * torch.pi * hz[:, None] * time
        sin_basis = torch.sin(angles)
        cos_basis = torch.cos(angles)
        first_sin = sin_basis
        first_cos = cos_basis
        second_angles = 2.0 * angles
        second_sin = torch.sin(second_angles)
        second_cos = torch.cos(second_angles)
        first = coeffs[:, 0:1] * first_sin + coeffs[:, 1:2] * first_cos
        second = coeffs[:, 2:3] * second_sin + coeffs[:, 3:4] * second_cos
        spectral = first + 0.5 * second
        return residual_gate * residual + gate * spectral


class TemporalHRHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hr_num_bins, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.in_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4)
        self.norm3 = nn.GroupNorm(groups, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hr_num_bins),
        )

    def _residual_block(self, x, conv, norm):
        y = conv(x)
        y = norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y

    def forward(self, z_seq):
        x = z_seq.permute(0, 2, 1)
        x = self.in_proj(x)
        x = self._residual_block(x, self.conv1, self.norm1)
        x = self._residual_block(x, self.conv2, self.norm2)
        x = self._residual_block(x, self.conv3, self.norm3)

        attention = torch.softmax(self.attention(x), dim=-1)
        weighted = torch.sum(x * attention, dim=-1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=-1),
                torch.std(x, dim=-1, unbiased=False),
                torch.amax(x, dim=-1),
            ],
            dim=1,
        )
        return self.classifier(pooled)


class SpectralTemporalHRHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        hr_num_bins,
        hr_low_bpm=45.0,
        hr_high_bpm=150.0,
        fs=30.0,
        dropout=0.1,
    ):
        super().__init__()
        self.hr_num_bins = int(hr_num_bins)
        self.hr_low_bpm = float(hr_low_bpm)
        self.hr_high_bpm = float(hr_high_bpm)
        self.fs = float(fs)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.in_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (5, 2), (7, 4), (7, 8)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.context_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hr_num_bins),
        )
        self.spectral_norm = nn.LayerNorm(hr_num_bins)
        self.spectral_scale = nn.Parameter(torch.tensor(1.0))
        self.spectral_bias = nn.Parameter(torch.zeros(hr_num_bins))

    def forward(self, z_seq):
        x = self.in_proj(z_seq.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        x = x - torch.mean(x, dim=-1, keepdim=True)

        frames = x.shape[-1]
        time = torch.arange(frames, device=x.device, dtype=x.dtype) / self.fs
        bpm = torch.linspace(self.hr_low_bpm, self.hr_high_bpm, self.hr_num_bins, device=x.device, dtype=x.dtype)
        angles = 2.0 * torch.pi * (bpm[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bct,kt->bck", x, sin_basis)
        cos_score = torch.einsum("bct,kt->bck", x, cos_basis)
        spectral_logits = torch.log((sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6)
        spectral_logits = self.spectral_norm(spectral_logits)

        pooled = torch.cat(
            [
                torch.mean(x, dim=-1),
                torch.std(x, dim=-1, unbiased=False),
                torch.amax(x, dim=-1),
                torch.amin(x, dim=-1),
            ],
            dim=1,
        )
        context_logits = self.context_classifier(pooled)
        return context_logits + self.spectral_scale * spectral_logits + self.spectral_bias


class TemporalScalarHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1, zero_init=False):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (7, 2), (9, 4), (11, 8)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4 if hidden_dim % 4 == 0 else 1,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention = nn.Linear(hidden_dim, 1)
        out = nn.Linear(hidden_dim, 1)
        if zero_init:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            out,
        )

    def forward(self, z_seq):
        x = self.input_proj(self.input_norm(z_seq))
        y = x.permute(0, 2, 1)
        for block in self.blocks:
            y = y + block(y)
        x = self.context(y.permute(0, 2, 1))
        weights = torch.softmax(self.attention(x), dim=1)
        weighted = torch.sum(x * weights, dim=1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=1),
                torch.std(x, dim=1, unbiased=False),
                torch.amax(x, dim=1),
                torch.amin(x, dim=1),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(-1)


class ResidualTemporalScalarHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.pooled_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_residual = TemporalScalarHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            zero_init=True,
        )
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z_seq, pooled_features):
        base = self.pooled_head(pooled_features).squeeze(-1)
        residual = self.temporal_residual(z_seq)
        return base + torch.sigmoid(self.residual_gate) * residual


class BandlimitedRateScalarHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        low_bpm=6.0,
        high_bpm=30.0,
        center=16.0,
        scale=8.0,
        fs=30.0,
        num_bins=96,
        dropout=0.1,
    ):
        super().__init__()
        self.low_bpm = float(low_bpm)
        self.high_bpm = float(high_bpm)
        self.center = float(center)
        self.scale = float(scale)
        self.fs = float(fs)
        self.num_bins = int(num_bins)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(7, 1), (11, 2), (15, 4), (21, 8), (31, 16)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.spectral_norm = nn.LayerNorm(self.num_bins)
        self.spectral_scale = nn.Parameter(torch.tensor(1.0))
        self.spectral_bias = nn.Parameter(torch.zeros(self.num_bins))
        self.context_gate = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(torch.tensor(1.0))
        self.context_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )
        residual_out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(residual_out.weight)
        nn.init.zeros_(residual_out.bias)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            residual_out,
        )
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z_seq):
        x = self.input_proj(self.input_norm(z_seq))
        y = x.permute(0, 2, 1)
        for block in self.blocks:
            y = y + block(y)
        y = y - torch.mean(y, dim=-1, keepdim=True)

        frames = y.shape[-1]
        time = torch.arange(frames, device=y.device, dtype=y.dtype) / self.fs
        bpm_bins = torch.linspace(
            self.low_bpm,
            self.high_bpm,
            self.num_bins,
            device=y.device,
            dtype=y.dtype,
        )
        angles = 2.0 * torch.pi * (bpm_bins[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bct,kt->bck", y, sin_basis)
        cos_score = torch.einsum("bct,kt->bck", y, cos_basis)
        spectral_logits = torch.log((sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6)
        spectral_logits = self.spectral_norm(spectral_logits) * self.spectral_scale + self.spectral_bias

        x_seq = y.permute(0, 2, 1)
        pooled = torch.cat(
            [
                torch.mean(x_seq, dim=1),
                torch.std(x_seq, dim=1, unbiased=False),
                torch.amax(x_seq, dim=1),
                torch.amin(x_seq, dim=1),
            ],
            dim=1,
        )
        logits = spectral_logits + torch.tanh(self.context_gate) * self.context_head(pooled)
        temperature = torch.clamp(F.softplus(self.log_temperature) + 1.0, min=1.0, max=25.0)
        weights = torch.softmax(logits * temperature, dim=1)
        rate_bpm = torch.sum(weights * bpm_bins.unsqueeze(0), dim=1)
        residual = self.residual_head(pooled).squeeze(-1)
        return (rate_bpm - self.center) / self.scale + torch.tanh(self.residual_gate) * residual


class VideoBandlimitedRateHead(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        low_bpm=6.0,
        high_bpm=30.0,
        center=16.0,
        scale=8.0,
        fs=30.0,
        num_bins=96,
        dropout=0.1,
    ):
        super().__init__()
        self.low_bpm = float(low_bpm)
        self.high_bpm = float(high_bpm)
        self.center = float(center)
        self.scale = float(scale)
        self.fs = float(fs)
        self.num_bins = int(num_bins)
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_proj = nn.Conv1d(15, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(7, 1), (11, 2), (15, 4), (21, 8), (31, 16)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.spectral_norm = nn.LayerNorm(self.num_bins)
        self.spectral_scale = nn.Parameter(torch.tensor(1.0))
        self.spectral_bias = nn.Parameter(torch.zeros(self.num_bins))
        self.context_gate = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(torch.tensor(1.0))
        self.context_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )

    @staticmethod
    def _safe_zscore(x, dim=1, eps=1e-6):
        return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True, unbiased=False) + eps)

    def forward(self, video_clip):
        if video_clip.shape[1] != 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        rgb_mean = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_std = video_clip.std(dim=(-1, -2), unbiased=False).permute(0, 2, 1)
        rgb_delta = torch.zeros_like(rgb_mean)
        rgb_delta[:, 1:] = rgb_mean[:, 1:] - rgb_mean[:, :-1]
        std_delta = torch.zeros_like(rgb_std)
        std_delta[:, 1:] = rgb_std[:, 1:] - rgb_std[:, :-1]
        green = rgb_mean[:, :, 1:2]
        green_delta = rgb_delta[:, :, 1:2]
        motion_energy = torch.mean(torch.abs(video_clip[:, :, 1:] - video_clip[:, :, :-1]), dim=(1, 3, 4)).unsqueeze(-1)
        motion_energy = F.pad(motion_energy, (0, 0, 1, 0))
        branch_input = torch.cat(
            [
                self._safe_zscore(rgb_mean, dim=1),
                self._safe_zscore(rgb_delta, dim=1),
                self._safe_zscore(rgb_std, dim=1),
                self._safe_zscore(std_delta, dim=1),
                self._safe_zscore(green, dim=1),
                self._safe_zscore(green_delta, dim=1),
                self._safe_zscore(motion_energy, dim=1),
            ],
            dim=-1,
        )
        x = self.input_proj(branch_input.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        x = x - torch.mean(x, dim=-1, keepdim=True)

        frames = x.shape[-1]
        time = torch.arange(frames, device=x.device, dtype=x.dtype) / self.fs
        bpm_bins = torch.linspace(
            self.low_bpm,
            self.high_bpm,
            self.num_bins,
            device=x.device,
            dtype=x.dtype,
        )
        angles = 2.0 * torch.pi * (bpm_bins[:, None] / 60.0) * time[None, :]
        basis_scale = torch.sqrt(torch.clamp(time.new_tensor(frames / 2.0), min=1.0))
        sin_basis = torch.sin(angles) / basis_scale
        cos_basis = torch.cos(angles) / basis_scale
        sin_score = torch.einsum("bct,kt->bck", x, sin_basis)
        cos_score = torch.einsum("bct,kt->bck", x, cos_basis)
        spectral_logits = torch.log((sin_score.square() + cos_score.square()).mean(dim=1) + 1e-6)
        spectral_logits = self.spectral_norm(spectral_logits) * self.spectral_scale + self.spectral_bias
        x_seq = x.permute(0, 2, 1)
        pooled = torch.cat(
            [
                torch.mean(x_seq, dim=1),
                torch.std(x_seq, dim=1, unbiased=False),
                torch.amax(x_seq, dim=1),
                torch.amin(x_seq, dim=1),
            ],
            dim=1,
        )
        logits = spectral_logits + torch.tanh(self.context_gate) * self.context_head(pooled)
        temperature = torch.clamp(F.softplus(self.log_temperature) + 1.0, min=1.0, max=25.0)
        weights = torch.softmax(logits * temperature, dim=1)
        rate_bpm = torch.sum(weights * bpm_bins.unsqueeze(0), dim=1)
        return (rate_bpm - self.center) / self.scale


class RespirationLowFrequencyHead(nn.Module):
    def __init__(self, hidden_dim=96, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_proj = nn.Conv1d(15, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(9, 1), (15, 2), (21, 4), (31, 8), (31, 16)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _safe_zscore(x, dim=1, eps=1e-6):
        return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True, unbiased=False) + eps)

    def forward(self, video_clip):
        if video_clip.shape[1] != 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        rgb_mean = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_std = video_clip.std(dim=(-1, -2), unbiased=False).permute(0, 2, 1)
        rgb_baseline = rgb_mean.mean(dim=1, keepdim=True)
        rgb_norm = rgb_mean / (rgb_baseline + 1e-6) - 1.0
        rgb_delta = torch.zeros_like(rgb_norm)
        rgb_delta[:, 1:] = rgb_norm[:, 1:] - rgb_norm[:, :-1]
        std_delta = torch.zeros_like(rgb_std)
        std_delta[:, 1:] = rgb_std[:, 1:] - rgb_std[:, :-1]
        green = rgb_norm[:, :, 1:2]
        green_delta = rgb_delta[:, :, 1:2]
        motion_energy = torch.mean(torch.abs(video_clip[:, :, 1:] - video_clip[:, :, :-1]), dim=(1, 3, 4)).unsqueeze(-1)
        motion_energy = F.pad(motion_energy, (0, 0, 1, 0))
        branch_input = torch.cat(
            [
                self._safe_zscore(rgb_norm, dim=1),
                self._safe_zscore(rgb_delta, dim=1),
                self._safe_zscore(rgb_std, dim=1),
                self._safe_zscore(std_delta, dim=1),
                self._safe_zscore(green, dim=1),
                self._safe_zscore(green_delta, dim=1),
                self._safe_zscore(motion_energy, dim=1),
            ],
            dim=-1,
        )
        x = self.input_proj(branch_input.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        weighted = torch.sum(x * weights, dim=-1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=-1),
                torch.std(x, dim=-1, unbiased=False),
                torch.amax(x, dim=-1),
                torch.amin(x, dim=-1),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(-1)


class ColorMotionScalarResidualHead(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.input_proj = nn.Conv1d(10, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for kernel_size, dilation in [(5, 1), (9, 2), (15, 4)]:
            padding = (kernel_size // 2) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            out,
        )

    @staticmethod
    def _safe_zscore(x, dim=1, eps=1e-6):
        return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True, unbiased=False) + eps)

    def forward(self, video_clip):
        if video_clip.shape[1] != 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        rgb_mean = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_std = video_clip.std(dim=(-1, -2), unbiased=False).permute(0, 2, 1)
        rgb_delta = torch.zeros_like(rgb_mean)
        rgb_delta[:, 1:] = rgb_mean[:, 1:] - rgb_mean[:, :-1]
        motion = torch.mean(torch.abs(video_clip[:, :, 1:] - video_clip[:, :, :-1]), dim=(1, 3, 4)).unsqueeze(-1)
        motion = F.pad(motion, (0, 0, 1, 0))
        branch_input = torch.cat(
            [
                self._safe_zscore(rgb_mean, dim=1),
                self._safe_zscore(rgb_delta, dim=1),
                self._safe_zscore(rgb_std, dim=1),
                self._safe_zscore(motion, dim=1),
            ],
            dim=-1,
        )
        x = self.input_proj(branch_input.permute(0, 2, 1))
        for block in self.blocks:
            x = x + block(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        weighted = torch.sum(x * weights, dim=-1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=-1),
                torch.std(x, dim=-1, unbiased=False),
                torch.amax(x, dim=-1),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(-1)


class RGBPOSFusionBranch(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        branch_hidden = max(hidden_dim // 2, 32)
        groups = 8 if branch_hidden % 8 == 0 else 1
        self.encoder = nn.Sequential(
            nn.Conv1d(8, branch_hidden, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, branch_hidden),
            nn.GELU(),
            nn.Conv1d(branch_hidden, branch_hidden, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.GroupNorm(groups, branch_hidden),
            nn.GELU(),
            nn.Conv1d(branch_hidden, feature_dim, kernel_size=1),
        )

    @staticmethod
    def _safe_zscore(signal, eps=1e-6):
        mean = torch.mean(signal, dim=1, keepdim=True)
        std = torch.std(signal, dim=1, keepdim=True, unbiased=False)
        return (signal - mean) / (std + eps)

    def forward(self, raw_video):
        rgb = raw_video.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_mean = rgb.mean(dim=1, keepdim=True)
        rgb_norm = rgb / (rgb_mean + 1e-6) - 1.0

        rgb_delta = torch.zeros_like(rgb_norm)
        rgb_delta[:, 1:] = rgb_norm[:, 1:] - rgb_norm[:, :-1]

        red = rgb_norm[:, :, 0]
        green = rgb_norm[:, :, 1]
        blue = rgb_norm[:, :, 2]
        x = green - blue
        y = green + blue - 2.0 * red
        alpha = torch.std(x, dim=1, keepdim=True, unbiased=False) / (
            torch.std(y, dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        pos = self._safe_zscore(x + alpha * y)
        pos_delta = torch.zeros_like(pos)
        pos_delta[:, 1:] = pos[:, 1:] - pos[:, :-1]

        branch_input = torch.cat(
            [
                rgb_norm,
                rgb_delta,
                pos.unsqueeze(-1),
                pos_delta.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.encoder(branch_input.permute(0, 2, 1)).permute(0, 2, 1)


class EncoderColorMotionBranch(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        branch_hidden = max(hidden_dim // 2, 32)
        groups = 8 if branch_hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(12, branch_hidden, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, branch_hidden),
            nn.GELU(),
            nn.Conv1d(branch_hidden, branch_hidden, kernel_size=7, padding=6, dilation=2, groups=groups, bias=False),
            nn.GroupNorm(groups, branch_hidden),
            nn.GELU(),
            nn.Conv1d(branch_hidden, branch_hidden, kernel_size=9, padding=16, dilation=4, groups=groups, bias=False),
            nn.GroupNorm(groups, branch_hidden),
            nn.GELU(),
            nn.Conv1d(branch_hidden, feature_dim, kernel_size=1),
        )

    @staticmethod
    def _safe_zscore(x, dim=1, eps=1e-6):
        return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True, unbiased=False) + eps)

    def forward(self, video_clip):
        rgb_mean = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_std = video_clip.std(dim=(-1, -2), unbiased=False).permute(0, 2, 1)
        rgb_baseline = rgb_mean.mean(dim=1, keepdim=True)
        rgb_norm = rgb_mean / (rgb_baseline + 1e-6) - 1.0
        rgb_norm = self._safe_zscore(rgb_norm, dim=1)

        rgb_delta = torch.zeros_like(rgb_norm)
        rgb_delta[:, 1:] = rgb_norm[:, 1:] - rgb_norm[:, :-1]
        std_delta = torch.zeros_like(rgb_std)
        std_delta[:, 1:] = rgb_std[:, 1:] - rgb_std[:, :-1]

        branch_input = torch.cat(
            [
                rgb_norm,
                self._safe_zscore(rgb_delta, dim=1),
                self._safe_zscore(rgb_std, dim=1),
                self._safe_zscore(std_delta, dim=1),
            ],
            dim=-1,
        )
        return self.net(branch_input.permute(0, 2, 1)).permute(0, 2, 1)


class GatedTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv_linear = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.conv_gate = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        linear = self.conv_linear(x)
        gate = self.conv_gate(x)
        out = torch.tanh(linear) * torch.sigmoid(gate)
        out = self.dropout(out)
        residual = x if self.downsample is None else self.downsample(x)
        if residual.size(1) != out.size(1):
            residual = residual[:, :out.size(1), :]
        return self.relu(out + residual)


class GatedTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for layer_idx, out_channels in enumerate(num_channels):
            dilation_size = 2 ** layer_idx
            in_channels = input_size if layer_idx == 0 else num_channels[layer_idx - 1]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                GatedTemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, padding, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels[-1], output_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = self.final_conv(out)
        return out.permute(0, 2, 1)


class EpisodeContextPhysioMixer(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, num_tokens=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.shared_tokens = nn.Parameter(torch.randn(1, self.num_tokens, feature_dim) * 0.02)
        self.hr_tokens = nn.Parameter(torch.randn(1, self.num_tokens, feature_dim) * 0.02)
        self.rr_tokens = nn.Parameter(torch.randn(1, self.num_tokens, feature_dim) * 0.02)
        self.context_from_stats = nn.Sequential(
            nn.LayerNorm(feature_dim * 6),
            nn.Linear(feature_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 3),
        )
        self.token_refiner = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.shared_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.hr_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.rr_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        groups = max(1, feature_dim // 16)
        self.shared_local = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=groups),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
        )
        self.hr_local = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2, groups=groups),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
        )
        self.rr_local = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=9, padding=4, groups=groups),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
        )
        self.shared_gate = nn.Parameter(torch.tensor(-2.5))
        self.hr_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_gate = nn.Parameter(torch.tensor(-1.5))
        self.norm = nn.LayerNorm(feature_dim)

    @staticmethod
    def _stats(x):
        return torch.cat([torch.mean(x, dim=1), torch.std(x, dim=1, unbiased=False)], dim=-1)

    @staticmethod
    def _local(block, x):
        return block(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, z_seq, hr_seq, rr_seq):
        batch = z_seq.shape[0]
        stats = torch.cat([self._stats(z_seq), self._stats(hr_seq), self._stats(rr_seq)], dim=-1)
        context = self.context_from_stats(stats).view(batch, 3, -1)
        shared_tokens = self.shared_tokens.expand(batch, -1, -1) + context[:, 0:1]
        hr_tokens = self.hr_tokens.expand(batch, -1, -1) + context[:, 1:2]
        rr_tokens = self.rr_tokens.expand(batch, -1, -1) + context[:, 2:3]
        tokens = self.token_refiner(torch.cat([shared_tokens, hr_tokens, rr_tokens], dim=1))
        shared_ctx = tokens[:, : self.num_tokens]
        hr_ctx = tokens[:, self.num_tokens : 2 * self.num_tokens]
        rr_ctx = tokens[:, 2 * self.num_tokens :]
        shared_delta, _ = self.shared_attn(self.norm(z_seq), shared_ctx, shared_ctx, need_weights=False)
        hr_memory = torch.cat([hr_ctx, shared_ctx], dim=1)
        rr_memory = torch.cat([rr_ctx, hr_ctx], dim=1)
        hr_delta, _ = self.hr_attn(self.norm(hr_seq), hr_memory, hr_memory, need_weights=False)
        rr_delta, _ = self.rr_attn(self.norm(rr_seq), rr_memory, rr_memory, need_weights=False)
        z_seq = z_seq + torch.sigmoid(self.shared_gate) * (shared_delta + self._local(self.shared_local, z_seq))
        hr_seq = hr_seq + torch.sigmoid(self.hr_gate) * (hr_delta + self._local(self.hr_local, hr_seq))
        rr_seq = rr_seq + torch.sigmoid(self.rr_gate) * (rr_delta + self._local(self.rr_local, rr_seq))
        return z_seq, hr_seq, rr_seq


class TemporalPositionPhysioAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, fs=30.0, dropout=0.1):
        super().__init__()
        self.fs = float(fs)
        self.register_buffer(
            "period_seconds",
            torch.tensor([5.0, 10.0, 30.0, 60.0, 120.0, 300.0], dtype=torch.float32),
            persistent=False,
        )
        embed_dim = 2 * 6 + 3
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.hr_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.rr_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.shared_gate = nn.Parameter(torch.tensor(-2.5))
        self.hr_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_gate = nn.Parameter(torch.tensor(-1.5))

    def _time_features(self, batch_size, frames, frame_offsets, device, dtype):
        local = torch.arange(frames, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        if frame_offsets is None:
            offsets = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            offsets = frame_offsets.to(device=device, dtype=dtype).view(batch_size, 1)
        seconds = (local + offsets) / max(self.fs, 1e-6)
        periods = self.period_seconds.to(device=device, dtype=dtype).view(1, 1, -1)
        phase = 2.0 * math.pi * seconds.unsqueeze(-1) / periods
        slow = torch.stack(
            [
                torch.log1p(seconds) / math.log(1.0 + 600.0),
                seconds / 600.0,
                local / max(float(frames - 1), 1.0),
            ],
            dim=-1,
        )
        return torch.cat([torch.sin(phase), torch.cos(phase), slow], dim=-1)

    @staticmethod
    def _apply_modulation(x, params, gate):
        scale, shift = params.chunk(2, dim=-1)
        return x + torch.sigmoid(gate) * (torch.tanh(scale) * x + shift)

    def forward(self, z_seq, hr_seq, rr_seq, frame_offsets=None):
        batch_size, frames, _, = z_seq.shape
        time_features = self._time_features(batch_size, frames, frame_offsets, z_seq.device, z_seq.dtype)
        z_seq = self._apply_modulation(z_seq, self.shared_mlp(time_features), self.shared_gate)
        hr_seq = self._apply_modulation(hr_seq, self.hr_mlp(time_features), self.hr_gate)
        rr_seq = self._apply_modulation(rr_seq, self.rr_mlp(time_features), self.rr_gate)
        return z_seq, hr_seq, rr_seq


class RateQueryPhysioAdapter(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        num_bins=32,
        num_heads=4,
        dropout=0.1,
        hr_low_bpm=45.0,
        hr_high_bpm=180.0,
        rr_low_bpm=6.0,
        rr_high_bpm=45.0,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.last_rate_query_logits = None
        self.register_buffer("hr_centers", torch.linspace(float(hr_low_bpm), float(hr_high_bpm), self.num_bins), persistent=False)
        self.register_buffer("rr_centers", torch.linspace(float(rr_low_bpm), float(rr_high_bpm), self.num_bins), persistent=False)
        self.shared_tokens = nn.Parameter(torch.randn(self.num_bins, feature_dim) * 0.02)
        self.hr_tokens = nn.Parameter(torch.randn(self.num_bins, feature_dim) * 0.02)
        self.rr_tokens = nn.Parameter(torch.randn(self.num_bins, feature_dim) * 0.02)
        self.shared_query = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )
        self.hr_query = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )
        self.rr_query = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins),
        )
        self.shared_film = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim * 2))
        self.hr_film = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim * 2))
        self.rr_film = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim * 2))
        self.hr_cross = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.rr_cross = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.shared_gate = nn.Parameter(torch.tensor(-2.5))
        self.hr_gate = nn.Parameter(torch.tensor(-2.0))
        self.rr_gate = nn.Parameter(torch.tensor(-1.5))
        self.norm = nn.LayerNorm(feature_dim)

    @staticmethod
    def _pool(x):
        return torch.cat(
            [
                torch.mean(x, dim=1),
                torch.std(x, dim=1, unbiased=False),
                torch.amax(x, dim=1),
                torch.amin(x, dim=1),
            ],
            dim=-1,
        )

    @staticmethod
    def _modulate(x, context, film, gate):
        scale, shift = film(context).unsqueeze(1).chunk(2, dim=-1)
        return x + torch.sigmoid(gate) * (torch.tanh(scale) * x + shift)

    def _context(self, x, query, tokens):
        logits = query(self._pool(x))
        weights = torch.softmax(logits, dim=-1)
        return weights @ tokens.to(device=x.device, dtype=x.dtype), logits

    def forward(self, z_seq, hr_seq, rr_seq):
        shared_context, _ = self._context(z_seq, self.shared_query, self.shared_tokens)
        hr_context, hr_logits = self._context(hr_seq, self.hr_query, self.hr_tokens)
        rr_context, rr_logits = self._context(rr_seq, self.rr_query, self.rr_tokens)
        self.last_rate_query_logits = {"hr": hr_logits, "rr": rr_logits}
        z_seq = self._modulate(z_seq, shared_context, self.shared_film, self.shared_gate)
        hr_memory = torch.stack([shared_context, hr_context], dim=1)
        rr_memory = torch.stack([shared_context, rr_context], dim=1)
        hr_delta, _ = self.hr_cross(self.norm(hr_seq), hr_memory, hr_memory, need_weights=False)
        rr_delta, _ = self.rr_cross(self.norm(rr_seq), rr_memory, rr_memory, need_weights=False)
        hr_seq = self._modulate(hr_seq + hr_delta, hr_context, self.hr_film, self.hr_gate)
        rr_seq = self._modulate(rr_seq + rr_delta, rr_context, self.rr_film, self.rr_gate)
        return z_seq, hr_seq, rr_seq


class PhaseNet(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        latent_dim=32,
        hidden_dim=128,
        tcn_layers=4,
        encoder_channels=(16, 32, 64, 128),
        encoder_expand_ratio=4,
        temporal_module="gated_tcn",
        phase_fs=30.0,
        phase_low_bpm=45.0,
        phase_high_bpm=150.0,
        phase_num_freq_bins=96,
        phase_dropout=0.1,
        physio_hr_low_bpm=45.0,
        physio_hr_high_bpm=150.0,
        physio_rr_low_bpm=6.0,
        physio_rr_high_bpm=30.0,
        physio_num_freq_bins=96,
        physio_long_context=False,
        hr_num_bins=0,
        hr_head_type="pooled",
        rgb_pos_fusion=False,
        encoder_input_normalization="none",
        long_range_mixer=False,
        waveform_head_type="gru",
        frequency_waveform_decoder=False,
        frequency_decoder_hr_low_bpm=45.0,
        frequency_decoder_hr_high_bpm=150.0,
        frequency_decoder_fs=30.0,
        encoder_motion_fusion=False,
        encoder_temporal_pyramid=False,
        encoder_color_motion_branch=False,
        mixstyle_stem=False,
        mixstyle_encoder=False,
        mixstyle_prob=0.5,
        mixstyle_alpha=0.1,
        dataset_conditioning=False,
        task_dataset_conditioning=False,
        dataset_num_domains=0,
        support_conditioning=False,
        support_context_dim=0,
        pos_waveform_branch=False,
        representation_refiner=False,
        representation_bottleneck_dim=32,
        vital_representation_adapter=False,
        cross_task_representation_adapter=False,
        rr_only_slow_adapter=False,
        prototype_scalar_readout=False,
        prototype_scalar_tokens=6,
        pre_temporal_physio_adapter=False,
        pre_temporal_physio_tokens=6,
        raw_motion_temporal_adapter=False,
        post_temporal_lowrank_adapter=False,
        post_temporal_lowrank_rank=16,
        hr_anchored_rr_adapter=False,
        hr_anchored_rr_rank=16,
        episodic_physio_whitening_adapter=False,
        episode_context_physio_mixer=False,
        episode_context_physio_tokens=6,
        temporal_position_physio_adapter=False,
        rate_query_physio_adapter=False,
        rate_query_physio_bins=32,
        task_conditioned_physio_norm=False,
        task_conditioned_physio_tokens=4,
        scalar_tasks=None,
        scalar_head_hidden_dim=None,
        scalar_head_type="pooled",
        rr_lowfreq_branch=False,
        rr_lowfreq_mode="replace",
        video_rate_branch=False,
        video_rate_mode="replace",
        color_scalar_branch=False,
        color_scalar_mode="residual",
        hr_band_residual_branch=False,
        rate_bin_aux=False,
        rate_bin_num_bins=96,
        rate_bin_scalar_mode="none",
        scalar_hr_low_bpm=45.0,
        scalar_hr_high_bpm=150.0,
        scalar_rr_low_bpm=6.0,
        scalar_rr_high_bpm=30.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        if len(self.encoder_channels) < 2:
            raise ValueError("PhaseNet encoder_channels must include stem and at least one encoder stage")
        self.encoder_expand_ratio = int(encoder_expand_ratio)
        self.temporal_module = str(temporal_module).lower()
        self.hr_num_bins = int(hr_num_bins)
        self.hr_head_type = str(hr_head_type).lower()
        self.rgb_pos_fusion = bool(rgb_pos_fusion)
        self.encoder_input_normalization = str(encoder_input_normalization).lower()
        self.encoder_motion_fusion = bool(encoder_motion_fusion)
        self.encoder_temporal_pyramid = bool(encoder_temporal_pyramid)
        self.encoder_color_motion_branch = bool(encoder_color_motion_branch)
        self.mixstyle_stem = MixStyle3D(p=mixstyle_prob, alpha=mixstyle_alpha) if bool(mixstyle_stem) else None
        self.mixstyle_encoder = MixStyle3D(p=mixstyle_prob, alpha=mixstyle_alpha) if bool(mixstyle_encoder) else None
        self.dataset_conditioning = (
            DatasetFiLM(int(dataset_num_domains), feature_dim) if bool(dataset_conditioning) and int(dataset_num_domains) > 0 else None
        )
        self.task_dataset_conditioning = (
            TaskSeparatedDatasetFiLM(int(dataset_num_domains), feature_dim)
            if bool(task_dataset_conditioning) and int(dataset_num_domains) > 0
            else None
        )
        self.support_conditioning = (
            SupportConditionedTaskFiLM(int(support_context_dim), feature_dim)
            if bool(support_conditioning) and int(support_context_dim) > 0
            else None
        )
        self.pos_waveform_branch = bool(pos_waveform_branch)
        self.representation_refiner_enabled = bool(representation_refiner)
        self.vital_representation_adapter_enabled = bool(vital_representation_adapter)
        self.cross_task_representation_adapter_enabled = bool(cross_task_representation_adapter)
        self.rr_only_slow_adapter_enabled = bool(rr_only_slow_adapter)
        self.prototype_scalar_readout_enabled = bool(prototype_scalar_readout)
        self.pre_temporal_physio_adapter_enabled = bool(pre_temporal_physio_adapter)
        self.raw_motion_temporal_adapter_enabled = bool(raw_motion_temporal_adapter)
        self.post_temporal_lowrank_adapter_enabled = bool(post_temporal_lowrank_adapter)
        self.hr_anchored_rr_adapter_enabled = bool(hr_anchored_rr_adapter)
        self.episodic_physio_whitening_adapter_enabled = bool(episodic_physio_whitening_adapter)
        self.episode_context_physio_mixer_enabled = bool(episode_context_physio_mixer)
        self.temporal_position_physio_adapter_enabled = bool(temporal_position_physio_adapter)
        self.rate_query_physio_adapter_enabled = bool(rate_query_physio_adapter)
        self.task_conditioned_physio_norm_enabled = bool(task_conditioned_physio_norm)
        self.scalar_tasks = tuple(scalar_tasks or ())
        self.rr_lowfreq_mode = str(rr_lowfreq_mode).lower()
        self.video_rate_mode = str(video_rate_mode).lower()
        self.color_scalar_mode = str(color_scalar_mode).lower()
        self.hr_band_residual_branch = bool(hr_band_residual_branch)
        self.rate_bin_aux = bool(rate_bin_aux)
        self.rate_bin_scalar_mode = str(rate_bin_scalar_mode).lower()
        self.scalar_hr_low_bpm = float(scalar_hr_low_bpm)
        self.scalar_hr_high_bpm = float(scalar_hr_high_bpm)
        self.scalar_rr_low_bpm = float(scalar_rr_low_bpm)
        self.scalar_rr_high_bpm = float(scalar_rr_high_bpm)
        stem_channels = self.encoder_channels[0]
        encoder_out_channels = self.encoder_channels[-1]
        self.stem = nn.Sequential(
            nn.Conv3d(3, stem_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.InstanceNorm3d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.motion_stem = None
        self.motion_gate = None
        if self.encoder_motion_fusion:
            self.motion_stem = nn.Sequential(
                nn.Conv3d(3, stem_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
                nn.InstanceNorm3d(stem_channels),
                nn.GELU(),
                nn.Conv3d(stem_channels, stem_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(stem_channels),
            )
            self.motion_gate = nn.Parameter(torch.tensor(0.0))
        encoder_blocks = []
        in_channels = stem_channels
        for out_channels in self.encoder_channels[1:]:
            encoder_blocks.append(
                EfficientSpatioTemporalBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=self.encoder_expand_ratio,
                )
            )
            in_channels = out_channels
        self.base_encoder = nn.Sequential(*encoder_blocks)
        self.temporal_pyramid = TemporalPyramidRefiner(encoder_out_channels) if self.encoder_temporal_pyramid else None
        self.temporal_pyramid_gate = nn.Parameter(torch.tensor(0.0)) if self.encoder_temporal_pyramid else None
        self.attention_head = SpatialAttentionHead(in_channels=encoder_out_channels)
        self.encoder_head = nn.Linear(encoder_out_channels, feature_dim)
        self.color_motion_branch = (
            EncoderColorMotionBranch(feature_dim, hidden_dim) if self.encoder_color_motion_branch else None
        )
        self.color_motion_gate = nn.Parameter(torch.tensor(0.0)) if self.encoder_color_motion_branch else None
        tcn_channels = [hidden_dim] * tcn_layers
        self.rgb_pos_branch = RGBPOSFusionBranch(feature_dim, hidden_dim) if self.rgb_pos_fusion else None
        temporal_input_dim = feature_dim * (3 if self.rgb_pos_fusion else 2)
        self.pre_temporal_physio_adapter = (
            PreTemporalPhysioAdapter(
                input_dim=temporal_input_dim,
                hidden_dim=hidden_dim,
                num_tokens=pre_temporal_physio_tokens,
                dropout=phase_dropout,
            )
            if self.pre_temporal_physio_adapter_enabled
            else None
        )
        self.raw_motion_temporal_adapter = (
            RawMotionTemporalAdapter(
                input_dim=temporal_input_dim,
                stats_dim=10,
                hidden_dim=hidden_dim,
                dropout=phase_dropout,
            )
            if self.raw_motion_temporal_adapter_enabled
            else None
        )
        if self.temporal_module in ("gated_tcn", "tcn", "default"):
            self.temporal_model = GatedTCN(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                num_channels=tcn_channels,
                kernel_size=3
            )
        elif self.temporal_module in ("phase_aware", "rppg_phase", "phase_temporal"):
            self.temporal_model = PhaseAwareTemporalMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                low_bpm=phase_low_bpm,
                high_bpm=phase_high_bpm,
                num_freq_bins=phase_num_freq_bins,
                dropout=phase_dropout,
            )
        elif self.temporal_module in ("physio_mixer", "physio", "physio_representation"):
            self.temporal_model = PhysioRepresentationMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("dual_rate_physio", "dual_physio", "dual_rate"):
            self.temporal_model = DualRatePhysioMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("task_token_physio", "task_token", "task_token_mixer"):
            self.temporal_model = TaskTokenPhysioMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("physio_oscillator", "oscillator", "phase_oscillator"):
            self.temporal_model = PhysioOscillatorMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("resp_envelope_physio", "resp_envelope", "resp_env"):
            self.temporal_model = RespEnvelopePhysioMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("spectral_token_physio", "spectral_token", "freq_token"):
            self.temporal_model = SpectralTokenPhysioMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("long_context_spectral_memory", "context_memory", "spectral_memory"):
            self.temporal_model = LongContextSpectralMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
            )
        elif self.temporal_module in ("dual_band_cross_attention", "dual_band_cross", "band_cross"):
            self.temporal_model = DualBandCrossAttentionMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("complex_dual_band_cross_attention", "complex_dual_band", "complex_band_cross"):
            self.temporal_model = ComplexDualBandCrossAttentionMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("complex_physio_prototype_memory", "prototype_memory", "proto_memory"):
            self.temporal_model = ComplexPhysioPrototypeMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("decoupled_physio_prototype_memory", "decoupled_proto_memory", "dppm"):
            self.temporal_model = DecoupledPhysioPrototypeMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("orthogonalized_physio_prototype_memory", "orthogonalized_proto_memory", "oppm"):
            self.temporal_model = OrthogonalizedPhysioPrototypeMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("cross_scale_cardioresp_decomposition", "cross_scale_decomp", "cscd"):
            self.temporal_model = CrossScaleCardioRespDecompositionMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_cardioresp_memory", "subharmonic_memory", "scrm"):
            self.temporal_model = SubharmonicCardioRespMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("lowfreq_motion_subharmonic", "lowfreq_motion_state", "lfms"):
            self.temporal_model = LowFreqMotionStateSubharmonicMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_rate_anchor_memory", "rate_anchor_subharmonic", "sram"):
            self.temporal_model = SubharmonicRateAnchorMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_cross_scale_memory", "subharmonic_cross_scale", "scsm"):
            self.temporal_model = SubharmonicCrossScaleMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_cross_scale_readout", "subharmonic_rr_readout", "scsr"):
            self.temporal_model = SubharmonicCrossScaleReadoutMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_envelope_residual", "subharmonic_envelope", "serm"):
            self.temporal_model = SubharmonicEnvelopeResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_envelope_rr_readout", "subharmonic_envelope_rr", "serr"):
            self.temporal_model = SubharmonicEnvelopeRRReadoutMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_instance_stable_residual", "subharmonic_instance_stable", "sisr"):
            self.temporal_model = SubharmonicInstanceStableResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_stream_norm_residual", "subharmonic_stream_norm", "ssnr"):
            self.temporal_model = SubharmonicStreamNormResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_task_basis_residual", "subharmonic_task_basis", "stbr"):
            self.temporal_model = SubharmonicTaskBasisResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_channel_calibration", "subharmonic_channel_calib", "sccm"):
            self.temporal_model = SubharmonicChannelCalibrationMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_state_residual", "subharmonic_state", "ssrm"):
            self.temporal_model = SubharmonicStateResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_temporal_filter_residual", "subharmonic_temporal_filter", "stfr"):
            self.temporal_model = SubharmonicTemporalFilterResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_dual_phase_residual", "subharmonic_dual_phase", "sdpr"):
            self.temporal_model = SubharmonicDualPhaseResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_parallel_state_fusion", "subharmonic_state_fusion", "spsf"):
            self.temporal_model = SubharmonicParallelStateFusionMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_rr_phase_residual", "subharmonic_rr_phase", "srpr"):
            self.temporal_model = SubharmonicRRPhaseResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_slow_rr_readout", "subharmonic_slow_readout", "ssro"):
            self.temporal_model = SubharmonicSlowRRReadoutMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("tri_prior_cardioresp_memory", "tri_prior_memory", "tpcm"):
            self.temporal_model = TriPriorCardioRespMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("adaptive_prior_cardioresp_memory", "adaptive_prior_memory", "apcm"):
            self.temporal_model = AdaptivePriorCardioRespMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("harmonic_frequency_operator", "frequency_operator", "freq_operator", "hfom"):
            self.temporal_model = HarmonicFrequencyOperatorMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("factorized_physio_subspace", "physio_subspace", "fpsm"):
            self.temporal_model = FactorizedPhysioSubspaceMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("style_factorized_physio_subspace", "style_physio_subspace", "sfpsm"):
            self.temporal_model = StyleAdaptiveFactorizedPhysioSubspaceMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("state_space_physio_memory", "physio_state_space", "sspm"):
            self.temporal_model = StateSpacePhysioMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("oscillator_subharmonic_hr_residual", "osc_subharmonic_hr", "oshr"):
            self.temporal_model = OscillatorSubharmonicHRResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("lowrank_cardioresp_residual", "lowrank_residual_memory", "lrrm"):
            self.temporal_model = LowRankCardioRespResidualMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("prototype_preserving_slow_rr", "proto_slow_rr", "ppsr"):
            self.temporal_model = PrototypePreservingSlowRRMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("subharmonic_slow_rr", "subharmonic_proto_slow_rr", "ssrr"):
            self.temporal_model = SubharmonicSlowRRMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("cardioresp_envelope_memory", "cardio_resp_envelope", "crem"):
            self.temporal_model = CardioRespEnvelopeMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("cardioresp_state_memory", "cardio_resp_state", "crsm"):
            self.temporal_model = CardioRespStateSpaceMemoryMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("continuous_cardioresp_state", "continuous_state", "ccrsm"):
            self.temporal_model = ContinuousCardioRespStateMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("adaptive_cardioresp_state", "adaptive_state", "acrsm"):
            self.temporal_model = AdaptiveCardioRespStateMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("latent_cardioresp_dynamics", "latent_dynamics", "lcdm"):
            self.temporal_model = LatentCardioRespDynamicsMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        elif self.temporal_module in ("hierarchical_cardioresp_dynamics", "hier_cardioresp_dynamics", "hcdm"):
            self.temporal_model = HierarchicalCardioRespDynamicsMixer(
                input_size=temporal_input_dim,
                output_size=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=tcn_layers,
                fs=phase_fs,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
                num_freq_bins=physio_num_freq_bins,
                dropout=phase_dropout,
                long_context=physio_long_context,
            )
        else:
            raise ValueError(f"Unsupported PhaseNet temporal module: {self.temporal_module}")
        self.temporal_refiner = LongRangeTemporalMixer(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=0.1,
        ) if bool(long_range_mixer) else None
        self.representation_refiner = (
            BottleneckTemporalRepresentationRefiner(
                feature_dim=feature_dim,
                bottleneck_dim=representation_bottleneck_dim,
                hidden_dim=hidden_dim,
                dropout=0.1,
            )
            if self.representation_refiner_enabled
            else None
        )
        self.vital_representation_adapter = (
            VitalBandRepresentationAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=0.1)
            if self.vital_representation_adapter_enabled
            else None
        )
        self.cross_task_representation_adapter = (
            CrossTaskPhysioAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=0.1)
            if self.cross_task_representation_adapter_enabled
            else None
        )
        self.rr_only_slow_adapter = (
            RROnlySlowAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=0.1)
            if self.rr_only_slow_adapter_enabled
            else None
        )
        self.post_temporal_lowrank_adapter = (
            PostTemporalLowRankAdapter(
                feature_dim=feature_dim,
                rank=post_temporal_lowrank_rank,
                dropout=phase_dropout,
            )
            if self.post_temporal_lowrank_adapter_enabled
            else None
        )
        self.hr_anchored_rr_adapter = (
            HRAnchoredRRResidualAdapter(
                feature_dim=feature_dim,
                rank=hr_anchored_rr_rank,
                dropout=phase_dropout,
            )
            if self.hr_anchored_rr_adapter_enabled
            else None
        )
        self.episodic_physio_whitening_adapter = (
            EpisodicPhysioWhiteningAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=phase_dropout)
            if self.episodic_physio_whitening_adapter_enabled
            else None
        )
        self.episode_context_physio_mixer = (
            EpisodeContextPhysioMixer(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_tokens=episode_context_physio_tokens,
                dropout=phase_dropout,
            )
            if self.episode_context_physio_mixer_enabled
            else None
        )
        self.temporal_position_physio_adapter = (
            TemporalPositionPhysioAdapter(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                fs=phase_fs,
                dropout=phase_dropout,
            )
            if self.temporal_position_physio_adapter_enabled
            else None
        )
        self.rate_query_physio_adapter = (
            RateQueryPhysioAdapter(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_bins=rate_query_physio_bins,
                dropout=phase_dropout,
                hr_low_bpm=physio_hr_low_bpm,
                hr_high_bpm=physio_hr_high_bpm,
                rr_low_bpm=physio_rr_low_bpm,
                rr_high_bpm=physio_rr_high_bpm,
            )
            if self.rate_query_physio_adapter_enabled
            else None
        )
        self.task_conditioned_physio_norm = (
            TaskConditionedPhysioNormMemory(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_tokens=task_conditioned_physio_tokens,
                dropout=phase_dropout,
            )
            if self.task_conditioned_physio_norm_enabled
            else None
        )
        self.projection_encoder = nn.Sequential(
            nn.Linear(feature_dim, (latent_dim + feature_dim) // 2),
            nn.ReLU(),
            nn.Linear((latent_dim + feature_dim) // 2, latent_dim)
        )
        self.decoder = Decoder1D(latent_dim=latent_dim, feature_dim=feature_dim)
        self.sim_alpha = 0.1
        self.waveform_head_type = str(waveform_head_type).lower()
        if self.waveform_head_type in ("multiscale", "conv", "long_range"):
            self.regressor_head = MultiScaleWaveformHead(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=0.1)
        elif self.waveform_head_type in ("gru", "sequence"):
            self.regressor_head = SequenceRegressor(feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported PhaseNet waveform head type: {self.waveform_head_type}")
        self.frequency_waveform_decoder = None
        self.pos_waveform_gate = nn.Parameter(torch.tensor(0.0)) if self.pos_waveform_branch else None
        if bool(frequency_waveform_decoder):
            if self.hr_num_bins <= 0:
                raise ValueError("frequency_waveform_decoder requires hr_num_bins > 0")
            self.frequency_waveform_decoder = FrequencyConditionedWaveformDecoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                hr_num_bins=self.hr_num_bins,
                hr_low_bpm=frequency_decoder_hr_low_bpm,
                hr_high_bpm=frequency_decoder_hr_high_bpm,
                fs=frequency_decoder_fs,
                dropout=0.1,
            )
        self.hr_spectral_head = None
        self.hr_spectral_gate = None
        if self.hr_head_type in ("temporal_regression", "regression"):
            self.hr_head = TemporalHRHead(feature_dim, hidden_dim, 1, dropout=0.1)
        elif self.hr_num_bins > 0:
            if self.hr_head_type in ("temporal", "sequence", "attention"):
                self.hr_head = TemporalHRHead(feature_dim, hidden_dim, self.hr_num_bins, dropout=0.1)
            elif self.hr_head_type in ("spectral", "neural_spectral", "spectral_temporal"):
                self.hr_head = SpectralTemporalHRHead(
                    feature_dim,
                    hidden_dim,
                    self.hr_num_bins,
                    hr_low_bpm=frequency_decoder_hr_low_bpm,
                    hr_high_bpm=frequency_decoder_hr_high_bpm,
                    fs=frequency_decoder_fs,
                    dropout=0.1,
                )
            elif self.hr_head_type in ("repr_bottleneck", "bottleneck", "bottleneck_temporal"):
                self.hr_head = BottleneckTemporalHRHead(
                    feature_dim,
                    hidden_dim,
                    self.hr_num_bins,
                    bottleneck_dim=representation_bottleneck_dim,
                    dropout=0.1,
                )
            elif self.hr_head_type in ("pooled_spectral", "pooled_plus_spectral", "residual_spectral"):
                self.hr_head = nn.Sequential(
                    nn.LayerNorm(feature_dim * 2),
                    nn.Linear(feature_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, self.hr_num_bins),
                )
                self.hr_spectral_head = SpectralTemporalHRHead(
                    feature_dim,
                    hidden_dim,
                    self.hr_num_bins,
                    hr_low_bpm=frequency_decoder_hr_low_bpm,
                    hr_high_bpm=frequency_decoder_hr_high_bpm,
                    fs=frequency_decoder_fs,
                    dropout=0.1,
                )
                self.hr_spectral_gate = nn.Parameter(torch.tensor(0.0))
            elif self.hr_head_type in ("pooled", "mean_std"):
                self.hr_head = nn.Sequential(
                    nn.LayerNorm(feature_dim * 2),
                    nn.Linear(feature_dim * 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, self.hr_num_bins),
                )
            else:
                raise ValueError(f"Unsupported PhaseNet HR head type: {self.hr_head_type}")
        else:
            self.hr_head = None
        scalar_hidden_dim = int(scalar_head_hidden_dim or hidden_dim)
        self.scalar_head_type = str(scalar_head_type).lower()
        self.scalar_heads = nn.ModuleDict()
        self.rate_residual_heads = nn.ModuleDict()
        self.rate_residual_gates = nn.ParameterDict()
        self.video_rate_heads = nn.ModuleDict()
        self.video_rate_gates = nn.ParameterDict()
        self.color_scalar_heads = nn.ModuleDict()
        self.color_scalar_gates = nn.ParameterDict()
        self.hr_band_residual_heads = nn.ModuleDict()
        self.hr_band_residual_gates = nn.ParameterDict()
        self.prototype_scalar_heads = nn.ModuleDict()
        self.prototype_scalar_gates = nn.ParameterDict()
        self.rate_bin_heads = nn.ModuleDict()
        self.rate_bin_scalar_gates = nn.ParameterDict()
        self.rate_bin_num_bins = int(rate_bin_num_bins)
        if self.prototype_scalar_readout_enabled:
            for task_name in ("hr", "rr"):
                if task_name in self.scalar_tasks:
                    self.prototype_scalar_heads[task_name] = PrototypeScalarReadout(
                        feature_dim=feature_dim,
                        hidden_dim=scalar_hidden_dim,
                        num_tokens=prototype_scalar_tokens,
                        dropout=0.1,
                    )
                    self.prototype_scalar_gates[task_name] = nn.Parameter(torch.tensor(-3.0))
        if self.rate_bin_aux and self.rate_bin_num_bins > 1:
            if "hr" in self.scalar_tasks:
                self.rate_bin_heads["hr"] = nn.Sequential(
                    nn.LayerNorm(feature_dim * 2),
                    nn.Linear(feature_dim * 2, scalar_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(scalar_hidden_dim, self.rate_bin_num_bins),
                )
                self.register_buffer(
                    "hr_rate_bin_centers",
                    torch.linspace(self.scalar_hr_low_bpm, self.scalar_hr_high_bpm, self.rate_bin_num_bins),
                    persistent=False,
                )
                if self.rate_bin_scalar_mode in ("blend", "gated", "residual", "add"):
                    self.rate_bin_scalar_gates["hr"] = nn.Parameter(torch.tensor(-2.0))
            if "rr" in self.scalar_tasks:
                self.rate_bin_heads["rr"] = nn.Sequential(
                    nn.LayerNorm(feature_dim * 2),
                    nn.Linear(feature_dim * 2, scalar_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(scalar_hidden_dim, self.rate_bin_num_bins),
                )
                self.register_buffer(
                    "rr_rate_bin_centers",
                    torch.linspace(self.scalar_rr_low_bpm, self.scalar_rr_high_bpm, self.rate_bin_num_bins),
                    persistent=False,
                )
                if self.rate_bin_scalar_mode in ("blend", "gated", "residual", "add"):
                    self.rate_bin_scalar_gates["rr"] = nn.Parameter(torch.tensor(-2.0))
        self.rr_lowfreq_head = (
            RespirationLowFrequencyHead(hidden_dim=max(64, scalar_hidden_dim), dropout=0.1)
            if bool(rr_lowfreq_branch) and "rr" in self.scalar_tasks
            else None
        )
        if bool(video_rate_branch):
            if "hr" in self.scalar_tasks:
                self.video_rate_heads["hr"] = VideoBandlimitedRateHead(
                    hidden_dim=scalar_hidden_dim,
                    low_bpm=self.scalar_hr_low_bpm,
                    high_bpm=self.scalar_hr_high_bpm,
                    center=80.0,
                    scale=30.0,
                    fs=30.0,
                    num_bins=128,
                    dropout=0.1,
                )
                self.video_rate_gates["hr"] = nn.Parameter(torch.tensor(0.0))
            if "rr" in self.scalar_tasks:
                self.video_rate_heads["rr"] = VideoBandlimitedRateHead(
                    hidden_dim=scalar_hidden_dim,
                    low_bpm=self.scalar_rr_low_bpm,
                    high_bpm=self.scalar_rr_high_bpm,
                    center=16.0,
                    scale=8.0,
                    fs=30.0,
                    num_bins=96,
                    dropout=0.1,
                )
                self.video_rate_gates["rr"] = nn.Parameter(torch.tensor(0.0))
        if bool(color_scalar_branch):
            for task_name in ("hr", "rr"):
                if task_name in self.scalar_tasks:
                    self.color_scalar_heads[task_name] = ColorMotionScalarResidualHead(
                        hidden_dim=max(32, scalar_hidden_dim // 2),
                        dropout=0.1,
                    )
                    self.color_scalar_gates[task_name] = nn.Parameter(torch.tensor(-4.0))
        if self.hr_band_residual_branch and "hr" in self.scalar_tasks:
            self.hr_band_residual_heads["hr"] = BandlimitedRateScalarHead(
                feature_dim=feature_dim,
                hidden_dim=scalar_hidden_dim,
                low_bpm=self.scalar_hr_low_bpm,
                high_bpm=self.scalar_hr_high_bpm,
                center=80.0,
                scale=30.0,
                fs=30.0,
                num_bins=128,
                dropout=0.1,
            )
            self.hr_band_residual_gates["hr"] = nn.Parameter(torch.tensor(-4.0))
        for task in self.scalar_tasks:
            task_name = str(task)
            if not task_name:
                raise ValueError("PhaseNet scalar task names must be non-empty")
            if self.scalar_head_type in (
                "pooled",
                "summary",
                "mlp",
                "pooled_plus_spectral_rate",
                "pooled_spectral_rate",
                "residual_spectral_rate",
            ):
                self.scalar_heads[task_name] = nn.Sequential(
                    nn.LayerNorm(feature_dim * 4),
                    nn.Linear(feature_dim * 4, scalar_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(scalar_hidden_dim, 1),
                )
                if self.scalar_head_type in (
                    "pooled_plus_spectral_rate",
                    "pooled_spectral_rate",
                    "residual_spectral_rate",
                ) and task_name in ("hr", "rr"):
                    if task_name == "hr":
                        self.rate_residual_heads[task_name] = BandlimitedRateScalarHead(
                            feature_dim=feature_dim,
                            hidden_dim=scalar_hidden_dim,
                            low_bpm=self.scalar_hr_low_bpm,
                            high_bpm=self.scalar_hr_high_bpm,
                            center=80.0,
                            scale=30.0,
                            fs=30.0,
                            num_bins=128,
                            dropout=0.1,
                        )
                    else:
                        self.rate_residual_heads[task_name] = BandlimitedRateScalarHead(
                            feature_dim=feature_dim,
                            hidden_dim=scalar_hidden_dim,
                            low_bpm=self.scalar_rr_low_bpm,
                            high_bpm=self.scalar_rr_high_bpm,
                            center=16.0,
                            scale=8.0,
                            fs=30.0,
                            num_bins=96,
                            dropout=0.1,
                        )
                    self.rate_residual_gates[task_name] = nn.Parameter(torch.tensor(-3.0))
            elif self.scalar_head_type in ("temporal", "attention", "temporal_attention"):
                self.scalar_heads[task_name] = TemporalScalarHead(
                    feature_dim=feature_dim,
                    hidden_dim=scalar_hidden_dim,
                    dropout=0.1,
                )
            elif self.scalar_head_type in ("residual_temporal", "temporal_residual"):
                self.scalar_heads[task_name] = ResidualTemporalScalarHead(
                    feature_dim=feature_dim,
                    hidden_dim=scalar_hidden_dim,
                    dropout=0.1,
                )
            elif self.scalar_head_type in ("hybrid_spectral", "spectral_rate", "bandlimited_rate"):
                if task_name == "hr":
                    self.scalar_heads[task_name] = BandlimitedRateScalarHead(
                        feature_dim=feature_dim,
                        hidden_dim=scalar_hidden_dim,
                        low_bpm=self.scalar_hr_low_bpm,
                        high_bpm=self.scalar_hr_high_bpm,
                        center=80.0,
                        scale=30.0,
                        fs=30.0,
                        num_bins=128,
                        dropout=0.1,
                    )
                elif task_name == "rr":
                    self.scalar_heads[task_name] = BandlimitedRateScalarHead(
                        feature_dim=feature_dim,
                        hidden_dim=scalar_hidden_dim,
                        low_bpm=self.scalar_rr_low_bpm,
                        high_bpm=self.scalar_rr_high_bpm,
                        center=16.0,
                        scale=8.0,
                        fs=30.0,
                        num_bins=96,
                        dropout=0.1,
                    )
                else:
                    self.scalar_heads[task_name] = nn.Sequential(
                        nn.LayerNorm(feature_dim * 4),
                        nn.Linear(feature_dim * 4, scalar_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(scalar_hidden_dim, 1),
                    )
            else:
                raise ValueError(f"Unsupported PhaseNet scalar head type: {self.scalar_head_type}")

    @staticmethod
    def _normalize_video_input(video_clip, normalization, eps=1e-6):
        if normalization in ("", "none", "raw"):
            return video_clip
        if video_clip.shape[1] == 3:
            reduce_dims = (2, 3, 4)
        elif video_clip.shape[2] == 3:
            reduce_dims = (1, 3, 4)
        else:
            raise ValueError(f"Cannot find channel dimension for input normalization: {video_clip.shape}")
        if normalization in ("channel_mean_center", "per_channel_mean_center"):
            channel_mean = video_clip.mean(dim=reduce_dims, keepdim=True)
            return video_clip / (channel_mean + eps) - 1.0
        if normalization in ("channel_zscore", "per_channel_zscore"):
            channel_mean = video_clip.mean(dim=reduce_dims, keepdim=True)
            channel_std = video_clip.std(dim=reduce_dims, keepdim=True, unbiased=False)
            return (video_clip - channel_mean) / (channel_std + eps)
        raise ValueError(f"Unsupported PhaseNet encoder input normalization: {normalization}")

    @staticmethod
    def _raw_motion_stats(video_clip):
        rgb_mean = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_std = video_clip.std(dim=(-1, -2), unbiased=False).permute(0, 2, 1)
        rgb_diff = torch.zeros_like(rgb_mean)
        rgb_diff[:, 1:] = rgb_mean[:, 1:] - rgb_mean[:, :-1]
        motion_energy = torch.mean(torch.abs(rgb_diff), dim=-1, keepdim=True)
        return torch.cat([rgb_mean, rgb_std, rgb_diff, motion_energy], dim=-1)

    def _forward_impl(
        self,
        video_clip,
        frame_offsets=None,
        domain_ids=None,
        support_context=None,
        return_features=False,
        return_scalar_outputs=False,
    ):
        if video_clip.dim() != 5:
            raise ValueError(f"Expected 5D input, got {video_clip.dim()}D")
        if video_clip.shape[1] == 3:
            pass
        elif video_clip.shape[2] == 3:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Invalid input shape: {video_clip.shape}. Channel dim must be 3.")

        batch_size, _, frames, _, _ = video_clip.shape
        encoder_video = self._normalize_video_input(video_clip, self.encoder_input_normalization)
        stem_features = self.stem(encoder_video)
        if self.motion_stem is not None:
            motion = torch.zeros_like(encoder_video)
            motion[:, :, 1:] = encoder_video[:, :, 1:] - encoder_video[:, :, :-1]
            stem_features = stem_features + torch.tanh(self.motion_gate) * self.motion_stem(motion)
        if self.mixstyle_stem is not None:
            stem_features = self.mixstyle_stem(stem_features)

        features = self.base_encoder(stem_features)
        if self.mixstyle_encoder is not None:
            features = self.mixstyle_encoder(features)
        if self.temporal_pyramid is not None:
            features = features + torch.tanh(self.temporal_pyramid_gate) * self.temporal_pyramid(features)
        attention_map = self.attention_head(features)
        features = features * attention_map
        features = torch.sum(features, dim=[-1, -2]).permute(0, 2, 1)
        z_raw = self.encoder_head(features)
        if self.color_motion_branch is not None:
            z_raw = z_raw + torch.tanh(self.color_motion_gate) * self.color_motion_branch(encoder_video)

        v_raw = torch.zeros_like(z_raw)
        v_raw[:, 1:] = z_raw[:, 1:] - z_raw[:, :-1]
        if self.rgb_pos_branch is not None:
            rgb_pos_features = self.rgb_pos_branch(video_clip)
            dynamic_features = torch.cat([z_raw, v_raw, rgb_pos_features], dim=-1)
        else:
            dynamic_features = torch.cat([z_raw, v_raw], dim=-1)
        if self.pre_temporal_physio_adapter is not None:
            dynamic_features = self.pre_temporal_physio_adapter(dynamic_features)
        if self.raw_motion_temporal_adapter is not None:
            dynamic_features = self.raw_motion_temporal_adapter(
                dynamic_features,
                self._raw_motion_stats(video_clip),
            )
        z_clean_seq = self.temporal_model(dynamic_features)
        hr_seq = getattr(self.temporal_model, "last_hr_seq", None)
        rr_seq = getattr(self.temporal_model, "last_rr_seq", None)
        oscillator_scalars = getattr(self.temporal_model, "last_oscillator_scalars", None)
        state_rate_bpm = getattr(self.temporal_model, "last_state_rate_bpm", None)
        if hr_seq is None:
            hr_seq = z_clean_seq
        if rr_seq is None:
            rr_seq = z_clean_seq
        if self.temporal_refiner is not None:
            z_clean_seq = self.temporal_refiner(z_clean_seq)
            hr_seq = self.temporal_refiner(hr_seq)
            rr_seq = self.temporal_refiner(rr_seq)
        if self.representation_refiner is not None:
            z_clean_seq = self.representation_refiner(z_clean_seq)
            hr_seq = self.representation_refiner(hr_seq)
            rr_seq = self.representation_refiner(rr_seq)
        if self.vital_representation_adapter is not None:
            z_clean_seq = self.vital_representation_adapter(z_clean_seq, task="shared")
            hr_seq = self.vital_representation_adapter(hr_seq, task="hr")
            rr_seq = self.vital_representation_adapter(rr_seq, task="rr")
        if self.cross_task_representation_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.cross_task_representation_adapter(z_clean_seq, hr_seq, rr_seq)
        if self.rr_only_slow_adapter is not None:
            rr_seq = self.rr_only_slow_adapter(rr_seq)
        if self.post_temporal_lowrank_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.post_temporal_lowrank_adapter(z_clean_seq, hr_seq, rr_seq)
        if self.hr_anchored_rr_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.hr_anchored_rr_adapter(z_clean_seq, hr_seq, rr_seq)
        if self.episodic_physio_whitening_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.episodic_physio_whitening_adapter(z_clean_seq, hr_seq, rr_seq)
        if self.episode_context_physio_mixer is not None:
            z_clean_seq, hr_seq, rr_seq = self.episode_context_physio_mixer(z_clean_seq, hr_seq, rr_seq)
        if self.temporal_position_physio_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.temporal_position_physio_adapter(
                z_clean_seq,
                hr_seq,
                rr_seq,
                frame_offsets=frame_offsets,
            )
        if self.rate_query_physio_adapter is not None:
            z_clean_seq, hr_seq, rr_seq = self.rate_query_physio_adapter(z_clean_seq, hr_seq, rr_seq)
        if self.task_conditioned_physio_norm is not None:
            z_clean_seq, hr_seq, rr_seq = self.task_conditioned_physio_norm(z_clean_seq, hr_seq, rr_seq)
        if self.support_conditioning is not None:
            z_clean_seq, hr_seq, rr_seq = self.support_conditioning(z_clean_seq, hr_seq, rr_seq, support_context)
        if self.task_dataset_conditioning is not None:
            z_clean_seq, hr_seq, rr_seq = self.task_dataset_conditioning(z_clean_seq, hr_seq, rr_seq, domain_ids)
        if self.dataset_conditioning is not None:
            z_clean_seq = self.dataset_conditioning(z_clean_seq, domain_ids)
            hr_seq = self.dataset_conditioning(hr_seq, domain_ids)
            rr_seq = self.dataset_conditioning(rr_seq, domain_ids)

        hr_logits = None
        if self.hr_head is not None:
            if self.hr_head_type in (
                "temporal",
                "sequence",
                "attention",
                "spectral",
                "neural_spectral",
                "spectral_temporal",
                "repr_bottleneck",
                "bottleneck",
                "bottleneck_temporal",
                "temporal_regression",
                "regression",
            ):
                hr_logits = self.hr_head(hr_seq)
            else:
                pooled = torch.cat(
                    [
                        torch.mean(hr_seq, dim=1),
                        torch.std(hr_seq, dim=1, unbiased=False),
                    ],
                    dim=-1,
                )
                hr_logits = self.hr_head(pooled)
                if self.hr_spectral_head is not None:
                    hr_logits = hr_logits + torch.tanh(self.hr_spectral_gate) * self.hr_spectral_head(hr_seq)

        scalar_outputs = None
        if self.scalar_heads:
            pooled_scalar_features = torch.cat(
                [
                    torch.mean(z_clean_seq, dim=1),
                    torch.std(z_clean_seq, dim=1, unbiased=False),
                    torch.amax(z_clean_seq, dim=1),
                    torch.amin(z_clean_seq, dim=1),
                ],
                dim=-1,
            )
            scalar_outputs = {}
            for task, head in self.scalar_heads.items():
                task_seq = rr_seq if task == "rr" else hr_seq if task in ("pr", "hr") else z_clean_seq
                task_pooled_features = torch.cat(
                    [
                        torch.mean(task_seq, dim=1),
                        torch.std(task_seq, dim=1, unbiased=False),
                        torch.amax(task_seq, dim=1),
                        torch.amin(task_seq, dim=1),
                    ],
                    dim=-1,
                )
                if isinstance(head, ResidualTemporalScalarHead):
                    scalar_outputs[task] = head(task_seq, task_pooled_features)
                elif isinstance(head, nn.Sequential):
                    scalar_outputs[task] = head(task_pooled_features).squeeze(-1)
                else:
                    scalar_outputs[task] = head(task_seq)
            if self.rate_query_physio_adapter is not None:
                rate_query_logits = getattr(self.rate_query_physio_adapter, "last_rate_query_logits", None)
                if rate_query_logits is not None:
                    if "hr" in rate_query_logits:
                        scalar_outputs["_hr_rate_query_logits"] = rate_query_logits["hr"]
                        scalar_outputs["_hr_rate_query_bins"] = self.rate_query_physio_adapter.hr_centers
                    if "rr" in rate_query_logits:
                        scalar_outputs["_rr_rate_query_logits"] = rate_query_logits["rr"]
                        scalar_outputs["_rr_rate_query_bins"] = self.rate_query_physio_adapter.rr_centers
            for task, head in self.rate_residual_heads.items():
                if task in scalar_outputs:
                    task_seq = rr_seq if task == "rr" else hr_seq if task in ("pr", "hr") else z_clean_seq
                    scalar_outputs[task] = scalar_outputs[task] + torch.sigmoid(self.rate_residual_gates[task]) * head(task_seq)
            for task, head in self.prototype_scalar_heads.items():
                if task in scalar_outputs:
                    task_seq = rr_seq if task == "rr" else hr_seq
                    scalar_outputs[task] = scalar_outputs[task] + torch.sigmoid(self.prototype_scalar_gates[task]) * head(task_seq)
            for task, head in self.video_rate_heads.items():
                video_rate = head(encoder_video)
                if self.video_rate_mode in ("replace", "only"):
                    scalar_outputs[task] = video_rate
                elif self.video_rate_mode in ("blend", "gated", "residual", "add"):
                    old_rate = scalar_outputs.get(task, video_rate)
                    gate = torch.sigmoid(self.video_rate_gates[task])
                    scalar_outputs[task] = old_rate + gate * (video_rate - old_rate)
                else:
                    raise ValueError(f"Unsupported video_rate_mode: {self.video_rate_mode}")
            for task, head in self.color_scalar_heads.items():
                color_scalar = head(encoder_video)
                if self.color_scalar_mode in ("replace", "only"):
                    scalar_outputs[task] = color_scalar
                elif self.color_scalar_mode in ("blend", "gated"):
                    old_scalar = scalar_outputs.get(task, color_scalar)
                    gate = torch.sigmoid(self.color_scalar_gates[task])
                    scalar_outputs[task] = old_scalar + gate * (color_scalar - old_scalar)
                elif self.color_scalar_mode in ("residual", "add"):
                    old_scalar = scalar_outputs.get(task, torch.zeros_like(color_scalar))
                    scalar_outputs[task] = old_scalar + torch.sigmoid(self.color_scalar_gates[task]) * color_scalar
                else:
                    raise ValueError(f"Unsupported color_scalar_mode: {self.color_scalar_mode}")
            for task, head in self.hr_band_residual_heads.items():
                if task in scalar_outputs:
                    scalar_outputs[task] = scalar_outputs[task] + torch.sigmoid(self.hr_band_residual_gates[task]) * head(hr_seq)
            for task, head in self.rate_bin_heads.items():
                task_seq = rr_seq if task == "rr" else hr_seq
                rate_features = torch.cat(
                    [
                        torch.mean(task_seq, dim=1),
                        torch.std(task_seq, dim=1, unbiased=False),
                    ],
                    dim=-1,
                )
                rate_logits = head(rate_features)
                rate_bins = getattr(self, f"{task}_rate_bin_centers")
                scalar_outputs[f"_{task}_rate_logits"] = rate_logits
                scalar_outputs[f"_{task}_rate_bins"] = rate_bins
                if self.rate_bin_scalar_mode not in ("", "none", "aux"):
                    expected_bpm = torch.sum(torch.softmax(rate_logits, dim=1) * rate_bins.to(rate_logits.device).unsqueeze(0), dim=1)
                    if task == "hr":
                        expected_scalar = (expected_bpm - 80.0) / 30.0
                    elif task == "rr":
                        expected_scalar = (expected_bpm - 16.0) / 8.0
                    else:
                        expected_scalar = None
                    if expected_scalar is not None:
                        if self.rate_bin_scalar_mode in ("replace", "only"):
                            scalar_outputs[task] = expected_scalar
                        elif self.rate_bin_scalar_mode in ("blend", "gated"):
                            old_scalar = scalar_outputs.get(task, expected_scalar)
                            gate = torch.sigmoid(self.rate_bin_scalar_gates[task])
                            scalar_outputs[task] = old_scalar + gate * (expected_scalar - old_scalar)
                        elif self.rate_bin_scalar_mode in ("residual", "add"):
                            old_scalar = scalar_outputs.get(task, torch.zeros_like(expected_scalar))
                            scalar_outputs[task] = old_scalar + torch.sigmoid(self.rate_bin_scalar_gates[task]) * expected_scalar
                        else:
                            raise ValueError(f"Unsupported rate_bin_scalar_mode: {self.rate_bin_scalar_mode}")
            if state_rate_bpm is not None:
                for task in ("hr", "rr"):
                    if task in state_rate_bpm:
                        scalar_outputs[f"_{task}_state_rate_bpm"] = state_rate_bpm[task]
            if self.rr_lowfreq_head is not None and "rr" in scalar_outputs:
                rr_lowfreq = self.rr_lowfreq_head(encoder_video)
                if self.rr_lowfreq_mode in ("replace", "only"):
                    scalar_outputs["rr"] = rr_lowfreq
                elif self.rr_lowfreq_mode in ("residual", "add"):
                    scalar_outputs["rr"] = scalar_outputs["rr"] + rr_lowfreq
                else:
                    raise ValueError(f"Unsupported rr_lowfreq_mode: {self.rr_lowfreq_mode}")
            if oscillator_scalars is not None:
                for task in ("pr", "hr", "rr"):
                    if task in scalar_outputs and task in oscillator_scalars:
                        scalar_outputs[task] = oscillator_scalars[task].to(device=scalar_outputs[task].device, dtype=scalar_outputs[task].dtype)

        if self.frequency_waveform_decoder is not None and hr_logits is not None:
            pred = self.frequency_waveform_decoder(z_clean_seq, hr_logits, frame_offsets=frame_offsets)
        else:
            pred = self.regressor_head(z_clean_seq)
        if self.pos_waveform_gate is not None:
            pred = pred + torch.tanh(self.pos_waveform_gate) * self._pos_waveform(video_clip)

        recon_loss = torch.tensor(0.0, device=video_clip.device)
        if self.training:
            z_clean_flat = z_clean_seq.reshape(batch_size * frames, -1)
            z_raw_flat = z_raw.reshape(batch_size * frames, -1)
            latent = self.projection_encoder(z_clean_flat)
            z_proj = self.decoder(latent)
            mse_loss = F.mse_loss(z_proj, z_raw_flat)
            cos_sim = F.cosine_similarity(z_proj, z_raw_flat, dim=-1).mean()
            recon_loss = mse_loss + self.sim_alpha * (1 - cos_sim)

        if return_features and return_scalar_outputs:
            return pred, recon_loss, hr_logits, z_clean_seq, z_raw, scalar_outputs
        if return_features:
            return pred, recon_loss, hr_logits, z_clean_seq, z_raw
        return pred, recon_loss, hr_logits

    @staticmethod
    def _pos_waveform(video_clip, eps=1e-6):
        rgb = video_clip.mean(dim=(-1, -2)).permute(0, 2, 1)
        rgb_mean = rgb.mean(dim=1, keepdim=True)
        rgb_norm = rgb / (rgb_mean + eps) - 1.0
        red = rgb_norm[:, :, 0]
        green = rgb_norm[:, :, 1]
        blue = rgb_norm[:, :, 2]
        x = green - blue
        y = green + blue - 2.0 * red
        alpha = torch.std(x, dim=1, keepdim=True, unbiased=False) / (
            torch.std(y, dim=1, keepdim=True, unbiased=False) + eps
        )
        pos = x + alpha * y
        return (pos - pos.mean(dim=1, keepdim=True)) / (pos.std(dim=1, keepdim=True, unbiased=False) + eps)

    def forward(self, video_clip, frame_offsets=None, domain_ids=None, support_context=None):
        pred, recon_loss, _ = self._forward_impl(
            video_clip,
            frame_offsets=frame_offsets,
            domain_ids=domain_ids,
            support_context=support_context,
        )
        return pred, recon_loss

    def forward_with_hr(self, video_clip, frame_offsets=None, domain_ids=None, support_context=None):
        return self._forward_impl(
            video_clip,
            frame_offsets=frame_offsets,
            domain_ids=domain_ids,
            support_context=support_context,
        )

    def forward_with_features(self, video_clip, frame_offsets=None, domain_ids=None, support_context=None):
        return self._forward_impl(
            video_clip,
            frame_offsets=frame_offsets,
            domain_ids=domain_ids,
            support_context=support_context,
            return_features=True,
        )

    def forward_with_multitask(self, video_clip, frame_offsets=None, domain_ids=None, support_context=None):
        pred, recon_loss, hr_logits, z_clean_seq, z_raw, scalar_outputs = self._forward_impl(
            video_clip,
            frame_offsets=frame_offsets,
            domain_ids=domain_ids,
            support_context=support_context,
            return_features=True,
            return_scalar_outputs=True,
        )
        return pred, recon_loss, hr_logits, scalar_outputs

    def forward_with_multitask_features(self, video_clip, frame_offsets=None, domain_ids=None, support_context=None):
        pred, recon_loss, hr_logits, z_clean_seq, z_raw, scalar_outputs = self._forward_impl(
            video_clip,
            frame_offsets=frame_offsets,
            domain_ids=domain_ids,
            support_context=support_context,
            return_features=True,
            return_scalar_outputs=True,
        )
        return pred, recon_loss, hr_logits, scalar_outputs, z_clean_seq, z_raw


class WindowContextHRHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hr_num_bins, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        token_dim = feature_dim * 2
        self.input_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hr_num_bins),
        )

    def forward(self, chunk_tokens):
        x = self.input_proj(chunk_tokens)
        x = self.context(x)
        weights = torch.softmax(self.attention(x), dim=1)
        weighted = torch.sum(x * weights, dim=1)
        pooled = torch.cat(
            [
                weighted,
                torch.mean(x, dim=1),
                torch.std(x, dim=1, unbiased=False),
                torch.amax(x, dim=1),
            ],
            dim=1,
        )
        return self.classifier(pooled)


class WindowContextPhaseNet(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        latent_dim=32,
        hidden_dim=128,
        tcn_layers=4,
        hr_num_bins=128,
        context_layers=2,
        context_heads=4,
        encoder_input_normalization="channel_mean_center",
    ):
        super().__init__()
        self.encoder_input_normalization = str(encoder_input_normalization).lower()
        self.base_encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            EfficientSpatioTemporalBlock(16, 32),
            EfficientSpatioTemporalBlock(32, 64),
            EfficientSpatioTemporalBlock(64, 128),
        )
        self.attention_head = SpatialAttentionHead(in_channels=128)
        self.encoder_head = nn.Linear(128, feature_dim)
        self.temporal_model = GatedTCN(
            input_size=feature_dim * 2,
            output_size=feature_dim,
            num_channels=[hidden_dim] * tcn_layers,
            kernel_size=3,
        )
        self.regressor_head = SequenceRegressor(feature_dim=feature_dim)
        self.hr_head = WindowContextHRHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            hr_num_bins=hr_num_bins,
            num_layers=context_layers,
            num_heads=context_heads,
        )

    @staticmethod
    def _normalize_video_input(video_clip, normalization, eps=1e-6):
        if normalization in ("", "none", "raw"):
            return video_clip
        if normalization in ("channel_mean_center", "per_channel_mean_center"):
            channel_mean = video_clip.mean(dim=(2, 3, 4), keepdim=True)
            return video_clip / (channel_mean + eps) - 1.0
        if normalization in ("channel_zscore", "per_channel_zscore"):
            channel_mean = video_clip.mean(dim=(2, 3, 4), keepdim=True)
            channel_std = video_clip.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
            return (video_clip - channel_mean) / (channel_std + eps)
        raise ValueError(f"Unsupported WindowContextPhaseNet normalization: {normalization}")

    def encode_clip(self, video_clip):
        video_clip = self._normalize_video_input(video_clip, self.encoder_input_normalization)
        features = self.base_encoder(video_clip)
        attention_map = self.attention_head(features)
        features = features * attention_map
        features = torch.sum(features, dim=[-1, -2]).permute(0, 2, 1)
        z_raw = self.encoder_head(features)
        v_raw = torch.zeros_like(z_raw)
        v_raw[:, 1:] = z_raw[:, 1:] - z_raw[:, :-1]
        z_clean_seq = self.temporal_model(torch.cat([z_raw, v_raw], dim=-1))
        ppg = self.regressor_head(z_clean_seq)
        token = torch.cat(
            [
                torch.mean(z_clean_seq, dim=1),
                torch.std(z_clean_seq, dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return token, ppg

    def forward(self, video_chunks):
        if video_chunks.dim() != 6:
            raise ValueError(f"Expected 6D window input B,N,C,T,H,W, got {video_chunks.dim()}D")
        batch_size, num_chunks, channels, frames, height, width = video_chunks.shape
        if channels != 3:
            raise ValueError(f"WindowContextPhaseNet expects channel-first chunks, got {video_chunks.shape}")
        flat = video_chunks.reshape(batch_size * num_chunks, channels, frames, height, width)
        tokens, ppg_chunks = self.encode_clip(flat)
        tokens = tokens.reshape(batch_size, num_chunks, -1)
        ppg = ppg_chunks.reshape(batch_size, num_chunks * frames)
        hr_logits = self.hr_head(tokens)
        return ppg, hr_logits
