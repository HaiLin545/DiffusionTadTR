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


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-3)
    return posemb


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


class TimeEmbedMLP(nn.Module):
    def __init__(self, d_model, scale_shift=True):
        super().__init__()

        self.scale_shift_embed = scale_shift

        if self.scale_shift_embed:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(4 * d_model, 2 * d_model)
            )
        else:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, tgt, time):
        if self.scale_shift_embed:
            scale_shift = self.time_mlp(time)
            scale, shift = scale_shift.chunk(2, dim=2)
            tgt = tgt * (scale + 1.0) + shift
        else:
            tgt = tgt + self.time_mlp(time)

        return tgt


class DeformableTransformer_old(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

        self.featToQueryEmbed = nn.Linear(2048, self.d_model)
        self.featToTgt = nn.Linear(2048,self.d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # shape=(bs)

    def forward(self, srcs, masks, pos_embeds, query_embed=None):

        assert query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2)
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(
            temporal_lens, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (temporal_lens.new_zeros((1,)), temporal_lens.cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1
        )  # (bs, nlevels)

        # deformable encoder
        memory = self.encoder(
            src_flatten,
            temporal_lens,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten if cfg.use_pos_embed else None,
            mask_flatten,
        )  # shape=(bs, t, c)

        bs, _, c = memory.shape


        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # (bs, Nq, c)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)                  # (bs, Nq, c) 
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            temporal_lens,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
        )
        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, memory.transpose(1, 2)

class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.memory_detach = cfg.memory_detach

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 1)
        self._reset_parameters()

        self.roi_size = 16
        self.query_proj = nn.Linear(self.roi_size * self.d_model, self.d_model)
        self.tgt_proj = nn.Linear(self.roi_size * self.d_model, self.d_model)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dynamic_conv = DynamicConv(d_model, roi_size=16)
        self.noise_embed = nn.Embedding(cfg.num_queries, d_model)
        self.no_query_embed = cfg.no_query_embed

        self.rn_before_dec = cfg.rn_before_dec
        if self.rn_before_dec:
            self.rcnn_head = RCNNHead(
                cfg, d_model, 20, dim_feedforward, nhead, dropout, activation
            )

        if self.no_query_embed:
            self.rcnn_head_1 = RCNNHead(
                cfg, d_model, 20, dim_feedforward, nhead, dropout, activation
            )
            self.rcnn_head_2 = RCNNHead(
                cfg, d_model, 20, dim_feedforward, nhead, dropout, activation
            )

        self.usc_af_backbone = cfg.usc_af_backbone
        if cfg.usc_af_backbone:
            self.backbone = ConvTransformerBackbone(
                n_in=d_model,
                n_embd=d_model,
                n_head=nhead,
                n_embd_ks=3,
                max_len=128,
                with_ln=True,
            )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # shape=(bs)

    def selfattn(self, feat):
        """
        feat: B, Nq, C
        """
        q = k = feat
        out = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), feat.transpose(0, 1)
        )[0].transpose(0, 1)
        out = out + self.dropout(out)
        out = self.norm(out)
        return out

    def forward(
        self, srcs, masks, pos_embeds, noise_segments, query_embed=None, time=None
    ):
        """
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
            time: (bs, )
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        """
        assert query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2)
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(
            temporal_lens, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (temporal_lens.new_zeros((1,)), temporal_lens.cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1
        )  # (bs, nlevels)

        time_embed = self.time_mlp(time).unsqueeze(1)

        # deformable encoder
        if not self.usc_af_backbone:
            memory = self.encoder(
                src_flatten,
                temporal_lens,
                level_start_index,
                valid_ratios,
                lvl_pos_embed_flatten if cfg.use_pos_embed else None,
                time_embed if cfg.use_enc_time_embed else None,
                mask_flatten,
            )  # shape=(bs, t, c)

        else:
            feat, mask = self.backbone(
                src_flatten.transpose(1, 2), ~mask_flatten.unsqueeze(1)
            )
            memory = feat[-1].transpose(1, 2)
            mask_flatten = ~mask[-1].squeeze(1)

        bs, _, c = memory.shape

        memory_new = memory.detach() if self.memory_detach else memory
        roi_src = memory_new if cfg.roi_with_memory else src_flatten
        roi_feat = self.get_embed_feat(
            roi_src, noise_segments
        )  # (bs, Nq, roi_size * c)

        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # (bs, Nq, c)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)  # (bs, Nq, c)

        if self.rn_before_dec:
            tgt = self.rcnn_head(roi_feat, tgt, time_embed)

        if self.no_query_embed:
            query_embed = self.rcnn_head_1(roi_feat, None, time_embed)
            tgt = self.rcnn_head_2(roi_feat, None, time_embed)

        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        # hs: (Ld, bs, Nq, c)
        # inter_references: (Ld, bs, Nq, 2) (ti, di)
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            temporal_lens,
            level_start_index,
            valid_ratios,
            query_embed,
            time_embed if cfg.use_dec_time_embed else None,
            mask_flatten,
            roi_feat if cfg.rcnnhead_dec else None,
        )
        inter_references_out = inter_references

        return (
            hs,
            init_reference_out,
            inter_references_out,
            memory.transpose(1, 2),
        )


class DiffusionDet(nn.Module):
    def __init__(
        self,
        nhead=8,
        in_dim=1024,
        d_model=256,
        num_classes=20,
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
            arch=(2, 4, 0),
            mha_win_size=[-1] * 1,
        )

        rcnn_head = RCNNHead(
            cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation
        )
        self.head_series = _get_clones(rcnn_head, num_decoder_layers)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

    def forward(self, srcs, masks, noise_segments, query_embed=None, time=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        for _, (src, mask) in enumerate(zip(srcs, masks)):
            src = src.transpose(1, 2)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        time_embed = self.time_mlp(time).unsqueeze(1)

        feat, mask = self.backbone(
            src_flatten.transpose(1, 2), ~mask_flatten.unsqueeze(1)
        )
        memory = feat[-1].transpose(1, 2)
        mask_flatten = ~mask[-1].squeeze(1)

        bs, _, c = memory.shape

        segment = noise_segments
        out_classes = []
        out_segments = []
        proposal_feat = None
        # roi_feat = self.get_embed_feat(memory, segment)

        for lid, layer in enumerate(self.head_series):
            out_class, out_seg, feat = layer(memory, segment, proposal_feat, time_embed)
            out_classes.append(out_class)
            out_segments.append(out_seg)

        return (
            torch.stack(out_classes),
            torch.stack(out_segments),
            memory.transpose(1, 2),
        )


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        time_embed,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        src2, _ = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if time_embed is not None:
            src = self.with_pos_embed(src, time_embed)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(
                0.5, T_ - 0.5, T_, dtype=torch.float32, device=device
            )  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)  # (bs, t)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = (
            reference_points[:, :, None] * valid_ratios[:, None]
        )  # (N, t, n_levels)
        return reference_points[..., None]  # (N, t, n_levels, 1)

    def forward(
        self,
        src,
        temporal_lens,
        level_start_index,
        valid_ratios,
        pos=None,
        time_embed=None,
        padding_mask=None,
    ):
        """
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        """
        output = src
        # (bs, t, levels, 1)
        reference_points = self.get_reference_points(
            temporal_lens, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                time_embed,
                reference_points,
                temporal_lens,
                level_start_index,
                padding_mask,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.rcnnhead_dec = cfg.rcnnhead_dec

        self.cross_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.Nq = cfg.num_queries
        self.time_embed = TimeEmbedMLP(d_model, cfg.scale_shift_embed)
        self.time_embed2 = TimeEmbedMLP(d_model, cfg.scale_shift_embed)
        self.time_embed3 = TimeEmbedMLP(d_model, cfg.scale_shift_embed)

        if self.rcnnhead_dec:
            self.rcnn_head = RCNNHead(
                cfg, d_model, 20, d_ffn, n_heads, dropout, activation
            )

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        time_embed,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        roi_feat=None,
    ):
        if not cfg.disable_query_self_att:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)

            tgt2 = self.self_attn(
                q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
            )[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        tgt2, _ = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if time_embed != None:
            tgt = self.time_embed(tgt, time_embed)

        if self.rcnnhead_dec:
            tgt = self.rcnn_head(roi_feat, tgt, time_embed)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.segment_embed = None
        self.class_embed = None

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        time_embed=None,
        src_padding_mask=None,
        roi_feat=None,
    ):
        """
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        """
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # (bs, nq, 1, 1 or 2) x (bs, 1, num_level, 1) => (bs, nq, num_level, 1 or 2)
            reference_points_input = (
                reference_points[:, :, None] * src_valid_ratios[:, None, :, None]
            )
            output = layer(
                output,
                query_pos,
                time_embed,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                roi_feat,
            )

            # hack implementation for segment refinement
            if self.segment_embed is not None:
                # update the reference point/segment of the next layer according to the output from the current layer
                tmp = self.segment_embed[lid](output)
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    # at the 0-th decoder layer
                    # d^(n+1) = delta_d^(n+1)
                    # c^(n+1) = sigmoid( inverse_sigmoid(c^(n)) + delta_c^(n+1))
                    assert reference_points.shape[-1] == 1
                    new_reference_points = tmp
                    new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            )

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(args):
    if not args.use_diffusiondet:
        return DeformableTransformer(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation,
            return_intermediate_dec=True,
            num_feature_levels=1,
            dec_n_points=args.dec_n_points,
            enc_n_points=args.enc_n_points,
        )
    else:
        return DiffusionDet(
            in_dim=2048,
            num_classes=20,
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.activation,
        )
