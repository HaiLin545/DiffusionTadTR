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

"""
TadTR model and criterion classes.
"""
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from opts import cfg
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    inverse_sigmoid,
)
from .schedule import linear_beta_schedule


if not cfg.disable_cuda:
    from models.ops.roi_align import ROIAlign

def get_norm(norm_type, dim, num_groups=None):
    if norm_type == "gn":
        assert num_groups is not None, "num_groups must be specified"
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == "bn":
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class DiffusionTadTR(TadTR):
    def __init__(
        self,
        position_embedding,
        transformer,
        num_classes,
        num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=True,
    ):
        super().__init__(
            position_embedding,
            transformer,
            num_classes,
            num_queries,
            aux_loss,
            with_segment_refine,
            with_act_reg,
        )
        self.transformer.get_embed_feat = self.get_embed_feat

    def get_embed_feat(self, feat, xt):
        """
        xt (bs, Nq, 2) (c,w)
        feat (bs, t, c)
        """
        bs, Nq = xt.shape[0], xt.shape[1]
        feat = feat.transpose(1, 2)
        rois = self._to_roi_align_format(xt, feat.shape[2], scale_factor=1.5)
        roi_features = self.roi_extractor(feat, rois)
        roi_features = roi_features.view((bs, Nq, -1, self.roi_size))
        return roi_features

    def forward(self, samples, noise_segments, t):
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask
        srcs = [self.input_proj[0](src)]
        masks = [mask]

        query_embeds = self.query_embed.weight

        outputs_classes = []
        outputs_coords = []

        hs, init_reference, inter_references, memory = self.transformer(
            srcs, masks, pos, noise_segments, query_embeds, t
        )

        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])

            tmp = self.segment_embed[lvl](hs[lvl])
            # the l-th layer (l >= 2)
            if reference.shape[-1] == 2:
                tmp += reference
            # the first layer
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if not self.with_act_reg:
            out = {"pred_logits": outputs_class[-1], "pred_segments": outputs_coord[-1]}
        else:
            # perform RoIAlign
            B, N = outputs_coord[-1].shape[:2]
            origin_feat = memory

            rois = self._to_roi_align_format(
                outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5
            )
            roi_features = self.roi_extractor(origin_feat, rois)
            roi_features = roi_features.view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)

            last_layer_cls = outputs_class[-1]
            last_layer_reg = outputs_coord[-1]

            out = {
                "pred_logits": last_layer_cls,
                "pred_segments": last_layer_reg,
                "pred_actionness": pred_actionness,
            }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

