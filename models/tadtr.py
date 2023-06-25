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
from models.matcher import build_matcher
from models.position_encoding import build_position_encoding
from .transformer import DeformableTransformer, TransformerForSparseRCNN
from .criterion import SetCriterion
from .postprocess import PostProcess
from .diffusion import DiffusionModel
from .rcnn.rcnn import SparseRCNNWrapper
from .rcnn.head import MLP


if not cfg.disable_cuda:
    from models.ops.roi_align import ROIAlign


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == "gn":
        assert num_groups is not None, "num_groups must be specified"
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == "bn":
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class TadTR(nn.Module):
    """This is the TadTR module that performs temporal action detection"""

    def __init__(
        self,
        position_embedding,
        num_classes,
        num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=True,
    ):
        super().__init__()
        self.num_queries = num_queries

        if cfg.rcnnWithTadtrEnc:
            self.transformer = TransformerForSparseRCNN(
                d_model=cfg.hidden_dim,
                nhead=cfg.nheads,
                num_encoder_layers=cfg.enc_layers,
                num_decoder_layers=cfg.dec_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.activation,
                return_intermediate_dec=True,
                num_feature_levels=1,
                dec_n_points=cfg.dec_n_points,
                enc_n_points=cfg.enc_n_points,
                num_classes=num_classes,
                num_queries=num_queries
            )
        else:
            self.transformer = DeformableTransformer(
                d_model=cfg.hidden_dim,
                nhead=cfg.nheads,
                num_encoder_layers=cfg.enc_layers,
                num_decoder_layers=cfg.dec_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.activation,
                return_intermediate_dec=True,
                num_feature_levels=1,
                dec_n_points=cfg.dec_n_points,
                enc_n_points=cfg.enc_n_points,
            )

        self.d_model = cfg.hidden_dim
        self.class_embed = nn.Linear(self.d_model, num_classes)
        self.segment_embed = MLP(self.d_model, self.d_model, 2, 3)
        self.query_embed = nn.Embedding(num_queries, self.d_model * 2)

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(2048, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ]
        )
        # self.backbone = backbone
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine
        self.with_act_reg = with_act_reg

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = cfg.dec_layers
        if with_segment_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.segment_embed = _get_clones(self.segment_embed, num_pred)
            nn.init.constant_(self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed
            self.transformer.segment_embed = self.segment_embed
        else:
            nn.init.constant_(self.segment_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.segment_embed = nn.ModuleList(
                [self.segment_embed for _ in range(num_pred)]
            )
            self.transformer.decoder.segment_embed = None

        if with_act_reg:
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

    def forward(self, samples):
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
        if cfg.rcnnWithTadtrEnc:
            outputs_class, outputs_coord, memory = self.transformer(
                srcs, masks, pos, query_embeds
            )
        else:
            hs, init_reference, inter_references, memory = self.transformer(
                srcs, masks, pos, query_embeds
            )

            outputs_classes = []
            outputs_coords = []
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

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_segments": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_name == "thumos14":
            num_classes = 20
        elif args.dataset_name == "muses":
            num_classes = 25
        elif args.dataset_name in ["activitynet", "hacs"]:
            num_classes = 200
        else:
            raise ValueError("unknown dataset {}".format(args.dataset_name))

    pos_embed = build_position_encoding(args)

    if args.use_sparse_rcnn:
        model = SparseRCNNWrapper(
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            with_segment_refine=args.seg_refine,
            with_act_reg=args.act_reg,
            position_embedding=pos_embed,
        )
    elif args.use_diffusion_det:
        model = DiffusionModel(
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            with_segment_refine=args.seg_refine,
            with_act_reg=args.act_reg,
        )
    else:
        model = TadTR(
            pos_embed,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            with_segment_refine=args.seg_refine,
            with_act_reg=args.act_reg,
        )

    matcher = build_matcher(args)
    losses = ["labels", "segments"]

    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_segments": args.seg_loss_coef,
        "loss_iou": args.iou_loss_coef,
    }

    if args.act_reg:
        weight_dict["loss_actionness"] = args.act_loss_coef
        losses.append("actionness")

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha
    )

    postprocessor = PostProcess()

    return model, criterion, postprocessor
