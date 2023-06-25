
import torch
import torch.nn.functional as F
from torch import nn

from opts import cfg
from .schedule import linear_beta_schedule
from .rcnn.rcnn import SparseRCNNWrapper




class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=True,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.timesteps = cfg.dm.timesteps  # cfg.DM.TIME_STEPS
        self.use_scale = cfg.dm.use_scale
        self.scale = cfg.dm.scale
        self.seg_renew = cfg.dm.use_seg_renew
        self.seg_renew_threshold = cfg.dm.seg_renew_threshold

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=self.timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        self.one_minus_alphas_bar = 1.0 - self.alphas_bar
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)
        self.one_minus_alphas_bar_prev = 1.0 - self.alphas_bar_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * self.one_minus_alphas_bar_prev / self.one_minus_alphas_bar
        )

        # ddim
        self.use_ddim = cfg.dm.use_ddim
        self.ddim_eta = cfg.dm.ddim_var_ratio
        self.ddim_steps = cfg.dm.ddim_step

        self.denoise_model = SparseRCNNWrapper(
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            with_segment_refine=with_segment_refine,
            with_act_reg=with_act_reg,
        )
        
    def pad_seg(self, seg, n):
        """
        input: seg (num_action, 2) (c,w)
        output: seg (N_train, 2)
        """
        if seg.shape[0]>n:
            return seg[:n]
        else:
            rand_seg = torch.randn((n - seg.shape[0],2),device=seg.device) / 6. + 0.5
            rand_seg[:,1] = torch.clip(rand_seg[:,1],min=1e-4)
            seg = torch.cat((seg, rand_seg))
            return seg

    def get_seg_start(self, targets, Nq = 40):
        segs = [self.pad_seg(t['segments'], Nq) for t in targets]
        return torch.stack(segs)

    def forward(self, samples, targets=None):

        if self.training:
            assert targets is not None
            gt_seg = self.get_seg_start(targets, Nq=self.num_queries)

            t = torch.randint(
                0, self.timesteps, (gt_seg.shape[0],), device=gt_seg.device
            ).long()

            if self.use_scale:
                gt_seg = (gt_seg * 2.0 - 1.0) * self.scale

            noise = torch.randn_like(gt_seg)
            x = self.q_sample(gt_seg, t, noise)

            if self.use_scale:
                x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
                noise_segments = ((x / self.scale) + 1) / 2.0
            else:
                noise_segments = x

            out = self.denoise_model(samples, noise_segments, t)

        else:
            out = self.ddim_sample_loop(samples)

        return out

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_bar_t = self.extract(self.sqrt_alphas_bar, t, x_start.shape)
        sqrt_one_minus_alphas_bar_t = self.extract(
            self.sqrt_one_minus_alphas_bar, t, x_start.shape
        )

        return sqrt_alphas_bar_t * x_start + sqrt_one_minus_alphas_bar_t * noise

    def segment_renew(self, segment, score):
        """
        segment: N, 2
        score: N, num_cls
        """
        N = segment.shape[0]
        score, _ = torch.max(score.sigmoid(), -1, keepdim=False)
        keep_idx = (score > self.seg_renew_threshold).bool()
        keep_num = torch.sum(keep_idx)
        segment = segment[keep_idx, :]
        if keep_num < N:
            rand_seg = (
                torch.randn((N - segment.shape[0], 2), device=segment.device) / 6.0
                + 0.5
            )
            rand_seg[:, 1] = torch.clip(rand_seg[:, 1], min=1e-4)
            segment = torch.cat((segment, rand_seg))
        return segment

    @torch.no_grad()
    def ddim_sample(self, samples, xt, t_prev, t_next):
        noise = torch.randn_like(xt)

        out = self.denoise_model(
            samples, xt, torch.full((xt.shape[0],), t_prev, device=xt.device)
        )

        if t_next < 0:
            return out
        else:
            alphas_bar_tprev = self.alphas_bar[t_prev]
            alphas_bar_tnext = self.alphas_bar[t_next]
            one_minus_alphas_bar_tprev = self.one_minus_alphas_bar[t_prev]
            one_minus_alphas_bar_tnext = self.one_minus_alphas_bar[t_next]
            sqrt_alphas_bar_tnext = self.sqrt_alphas_bar[t_next]

            x0, score = out["pred_segments"], out["pred_logits"]

            if self.seg_renew:
                x0 = torch.stack(
                    [self.segment_renew(seg, s) for (seg, s) in zip(x0, score)]
                )

            if self.use_scale:
                x0 = (x0 * 2.0 - 1.0) * self.scale
                x0 = torch.clamp(x0, min=-1 * self.scale, max=self.scale)

            pred_noise = (
                xt - x0 * self.sqrt_alphas_bar[t_prev]
            ) / self.sqrt_one_minus_alphas_bar[t_prev]

            var = (
                one_minus_alphas_bar_tnext
                / one_minus_alphas_bar_tprev
                * (1.0 - alphas_bar_tprev / alphas_bar_tnext)
            )
            sigma = self.ddim_eta * var.sqrt()
            mean = (
                sqrt_alphas_bar_tnext * x0
                + (one_minus_alphas_bar_tnext - sigma**2).sqrt() * pred_noise
            )
            segment = mean + sigma * noise
            out["pred_segments"] = segment
            return out

    @torch.no_grad()
    def ddim_sample_loop(self, samples):
        device = next(self.denoise_model.parameters()).device
        # start from pure noise (for each example in the batch)
        bs = samples[0].shape[0]
        segment_t = torch.randn((bs, self.num_queries, 2), device=device) / 6.0 + 0.5
        segment_t[..., 1] = torch.clip(segment_t[..., 1], min=1e-2)

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.timesteps - 1, steps=self.ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        segments = []

        for time, time_next in time_pairs:
            out = self.ddim_sample(samples, segment_t, time, time_next)
            segment_t = out["pred_segments"]
            segments.append(segment_t)
        return out  # t , batchsize, chennels, h, w

    @torch.no_grad()
    def infer(self, samples):
        out = self.ddim_sample_loop(samples)
        return out

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

