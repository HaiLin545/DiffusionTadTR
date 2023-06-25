# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy

import torch
import torch.nn.functional as F
import math
from opts import cfg
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.temporal_deform_attn import DeformAttn
from util.segment_ops import segment_cw_to_t1t2
from .head import RCNNHead
from .backbone import ConvTransformerBackbone
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    inverse_sigmoid,
)
from models.ops.roi_align import ROIAlign


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SparseRCNN(nn.Module):
    def __init__(
        self,
        nhead=8,
        in_dim=1024,
        d_model=256,
        num_classes=20,
        num_queries=40,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        self.backbone = ConvTransformerBackbone(
            n_in=in_dim,
            n_embd=d_model,
            n_head=nhead,
            n_embd_ks=3,
            max_len=128,
            with_ln=True,
            arch=(cfg.backbone_arch[0], cfg.backbone_arch[1], cfg.backbone_arch[2]),
            mha_win_size=[-1] * (cfg.backbone_arch[-1]+1),
        )

        self.return_intermediate = True
        self.num_classes = num_classes
        rcnn_head = RCNNHead(
            cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation
        )
        self.head_series = _get_clones(rcnn_head, num_decoder_layers)
        self.use_tadtr_enc = cfg.use_tadtr_enc
        self.proposal_feature = nn.Embedding(num_queries, d_model)
        self.proposal_segment = nn.Embedding(num_queries, 2)
        nn.init.constant_(self.proposal_segment.weight[:, 0], 0.5)
        nn.init.constant_(self.proposal_segment.weight[:, 1], 1.0)

        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

    def forward(self, srcs, masks, noise_segments=None, time=None):
        src_flatten = []
        mask_flatten = []

        for _, (src, mask) in enumerate(zip(srcs, masks)):
            src = src.transpose(1, 2)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)

        feat, mask = self.backbone(
            src_flatten.transpose(1, 2), ~mask_flatten.unsqueeze(1)
        )
        memory = feat[-1].transpose(1, 2)

        bs, _, c = memory.shape

        if noise_segments is not None:
            proposal_seg = noise_segments
            proposal_feat = None
        else:
            proposal_seg = self.proposal_segment.weight
            proposal_seg = proposal_seg.unsqueeze(0).expand(bs, -1, -1)
            
        proposal_feat = self.proposal_feature.weight
        proposal_feat = proposal_feat.unsqueeze(0).expand(bs, -1, -1)

        out_classes = []
        out_segments = []

        time_embed = self.time_mlp(time) if time is not None else None

        for lid, rcnn_head in enumerate(self.head_series):
            out_class, out_seg, feat = rcnn_head(
                memory, proposal_seg, proposal_feat, time_embed
            )

            out_classes.append(out_class)
            out_segments.append(out_seg)

            proposal_seg = out_seg.detach()

        if self.return_intermediate:
            return torch.stack(out_classes), torch.stack(out_segments), memory

        return out_classes[None], out_segments[None], memory


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SparseRCNNWrapper(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=True,
        position_embedding=None,
    ):
        super().__init__()
        self.transformer = SparseRCNN(
            nhead=cfg.nheads,
            in_dim=2048,
            d_model=cfg.hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=cfg.dec_layers,
            dim_feedforward=cfg.dim_feedforward,
            activation=cfg.activation,
            dropout=cfg.dropout,
        )
        self.with_act_reg = with_act_reg
        self.with_segment_refine = with_segment_refine
        self.aux_loss = aux_loss
        self.num_classes = num_classes
        self.num_querise = num_queries
        self.d_model = cfg.hidden_dim

        if self.with_act_reg:
            # RoIAlign params
            self.roi_size = cfg.roi_size
            self.roi_scale = 0
            self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
            self.actionness_pred = nn.Sequential(
                nn.Linear(self.roi_size * self.d_model, self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, 1),
                nn.Sigmoid(),
            )

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * scale_factor
        rois_abs = (
            torch.cat((rois_center - rois_size / 2, rois_center + rois_size / 2), dim=2)
            * T
        )
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((B * N, 3)).detach()

    def forward(self, samples, noise_segments=None, t=None, targets=None):
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        src, mask = samples.tensors, samples.mask
        srcs = [src]
        masks = [mask]

        outputs_class, outputs_coord, memory = self.transformer(
            srcs, masks, noise_segments, t
        )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_segments": outputs_coord[-1],
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.with_act_reg:
            B, N = outputs_coord[-1].shape[:2]
            origin_feat = memory.transpose(1, 2)

            rois = self._to_roi_align_format(
                outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5
            )
            roi_features = self.roi_extractor(origin_feat, rois)
            roi_features = roi_features.view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)

            out["pred_actionness"] = pred_actionness

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_segments": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
