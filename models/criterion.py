import torch
import torch.nn.functional as F
from torch import nn

from util import segment_ops
from util.misc import (
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
)

from .custom_loss import sigmoid_focal_loss


class SetCriterion(nn.Module):
    """This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_segments,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )  # nq
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
        targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
        The target segments are expected in format (center, width), normalized by the video length.
        """
        assert "pred_segments" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs["pred_segments"][idx]
        target_segments = torch.cat(
            [t["segments"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_segment = F.l1_loss(src_segments, target_segments, reduction="none")

        losses = {}
        losses["loss_segments"] = loss_segment.sum() / num_segments

        loss_iou = 1 - torch.diag(
            segment_ops.segment_iou(
                segment_ops.segment_cw_to_t1t2(src_segments),
                segment_ops.segment_cw_to_t1t2(target_segments),
            )
        )
        losses["loss_iou"] = loss_iou.sum() / num_segments
        return losses

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
        targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
        The target segments are expected in format (center, width), normalized by the video length.
        """
        assert "pred_segments" in outputs
        assert "pred_actionness" in outputs
        src_segments = outputs["pred_segments"].view((-1, 2))
        target_segments = torch.cat([t["segments"] for t in targets], dim=0)

        losses = {}

        iou_mat = segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments),
        )

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs["pred_actionness"]
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())

        losses["loss_actionness"] = loss_actionness
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "segments": self.loss_segments,
            "actionness": self.loss_actionness,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor(
            [num_segments],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # we do not compute actionness loss for aux outputs
                    if "actionness" in loss:
                        continue
                    if "noise" in loss:
                        continue

                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_segments, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.indices = indices
        return losses
