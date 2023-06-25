# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

from models.ops.roi_align import ROIAlign


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RCNNHead(nn.Module):
    def __init__(
        self,
        cfg,
        d_model,
        num_classes,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model * 4, d_model * 2)
        )
        self.use_tadtr_head = cfg.use_tadtr_head
        if cfg.use_tadtr_head:
            self.class_embed = nn.Linear(d_model, num_classes)
            self.segment_embed = MLP(d_model, d_model, 2, 3)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
            # nn.init.constant_(self.segment_embed.layers[-1].bias.data[1:], -2.0)

        num_cls = cfg.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 2)
        self.scale_clamp = math.log(100000.0 / 16)

        self.roi_size = cfg.roi_size
        self.roi_scale = 0
        self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
        self.no_activation = cfg.rcnn_head_no_activation

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        """Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        """
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

    # def forward(self, features, bboxes, pro_features, time_emb, roi_features=None):
    def forward(
        self,
        features,
        bboxes,
        pro_features,
        time_embed=None,
    ):
        """
        :param bboxes: (b, N, 2)
        :param pro_features: (b, N, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.

        origin_feat = features.transpose(1,2)
        rois = self._to_roi_align_format(
            bboxes, origin_feat.shape[2], scale_factor=1.5
        )
        roi_features = self.roi_extractor(origin_feat, rois)  # B * N, d, roi_size

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)       # B, N, d

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        # with time embed
        if time_embed is not None:
            scale_shift = self.block_time_mlp(time_embed)
            scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            fc_feature = fc_feature * (scale + 1) + shift

        if not self.use_tadtr_head:
            cls_feature = fc_feature.clone()
            reg_feature = fc_feature.clone()
            for cls_layer in self.cls_module:
                cls_feature = cls_layer(cls_feature)
            for reg_layer in self.reg_module:
                reg_feature = reg_layer(reg_feature)
            class_logits = self.class_logits(cls_feature)
            bboxes_deltas = self.bboxes_delta(reg_feature)#.sigmoid()
            pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.reshape(-1, 2))

        else:
            class_logits = self.class_embed(fc_feature)
            pred_bboxes = self.segment_embed(fc_feature).sigmoid()
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        gc = boxes[:, 0]
        gw = boxes[:, 1]
        dc = deltas[:, 0]
        dw = deltas[:, 1]

        # Prevent sending too large values into torch.exp()
        # dc = torch.clamp(dc, max=self.scale_clamp)
        dw = torch.clamp(dw, max=self.scale_clamp)

        pred_c = dc * gw + gc
        pred_w = torch.exp(dw) * gw

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = pred_c
        pred_boxes[:, 1] = pred_w

        return pred_boxes


class DynamicConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.hidden_dim
        self.dim_dynamic = cfg.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params) 

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        self.roi_size = cfg.roi_size
        num_output = self.hidden_dim * self.roi_size
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1,  B * Nq, self.d_model)
        roi_features: (roi_size, B * Nq, self.d_model)
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(
            -1, self.hidden_dim, self.dim_dynamic
        )
        param2 = parameters[:, :, self.num_params :].view(
            -1, self.dim_dynamic, self.hidden_dim
        )

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
