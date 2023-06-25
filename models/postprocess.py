import torch
from torch import nn

from opts import cfg
from util import segment_ops

class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, fuse_score=True):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs["pred_logits"], outputs["pred_segments"]

        assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 1

        prob = out_logits.sigmoid()  # [bs, nq, C]
        if fuse_score:
            prob *= outputs["pred_actionness"]

        segments = segment_ops.segment_cw_to_t1t2(out_segments)  # bs, nq, 2

        if cfg.postproc_rank == 1:  # default
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(
                prob.view(out_logits.shape[0], -1),
                min(cfg.postproc_ins_topk, prob.shape[1] * prob.shape[2]),
                dim=1,
            )
            scores = topk_values
            topk_segments = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]

            # bs, nq, 2; bs, num, 2
            segments = torch.gather(
                segments, 1, topk_segments.unsqueeze(-1).repeat(1, 1, 2)
            )
            query_ids = topk_segments
        else:
            # pick topk classes for each query
            # pdb.set_trace()
            scores, labels = torch.topk(prob, cfg.postproc_cls_topk, dim=-1)
            scores, labels = scores.flatten(1), labels.flatten(1)
            # (bs, nq, 1, 2)
            segments = segments[
                :,
                [
                    i // cfg.postproc_cls_topk
                    for i in range(cfg.postproc_cls_topk * segments.shape[1])
                ],
                :,
            ]
            query_ids = (
                torch.arange(
                    0,
                    cfg.postproc_cls_topk * segments.shape[1],
                    1,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                // cfg.postproc_cls_topk
            )[None, :].repeat(labels.shape[0], 1)

        # from normalized [0, 1] to absolute [0, length] coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1)
        segments = segments * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "segments": b, "query_ids": q}
            for s, l, b, q in zip(scores, labels, segments, query_ids)
        ]

        return results
