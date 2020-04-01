import torch
from torch import nn

from smoke.modeling.smoke_coder import SMOKECoder
from smoke.layers.utils import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)


class PostProcessor(nn.Module):
    def __init__(self,
                 smoker_coder,
                 reg_head,
                 det_threshold,
                 max_detection,
                 pred_2d):
        super(PostProcessor, self).__init__()
        self.smoke_coder = smoker_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d

    def prepare_targets(self, targets):
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        size = torch.stack([torch.tensor(t.size) for t in targets])

        return dict(trans_mat=trans_mat,
                    K=K,
                    size=size)

    def forward(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch = pred_heatmap.shape[0]

        target_varibales = self.prepare_targets(targets)

        heatmap = nms_hm(pred_heatmap)

        scores, indexs, clses, ys, xs = select_topk(
            heatmap,
            K=self.max_detection,
        )

        pred_regression = select_point_of_interest(
            batch, indexs, pred_regression
        )
        pred_regression_pois = pred_regression.view(-1, self.reg_head)

        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            target_varibales["K"],
            target_varibales["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )

        if self.pred_2d:
            box2d = self.smoke_coder.encode_box2d(
                target_varibales["K"],
                pred_rotys,
                pred_dimensions,
                pred_locations,
                target_varibales["size"]
            )
        else:
            box2d = torch.tensor([0, 0, 0, 0])

        # change variables to the same dimension
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

        result = torch.cat([
            clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores
        ], dim=1)

        keep_idx = result[:, -1] > self.det_threshold
        result = result[keep_idx]

        return result


def make_smoke_post_processor(cfg):
    smoke_coder = SMOKECoder(
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        cfg.MODEL.DEVICE,
    )

    postprocessor = PostProcessor(
        smoke_coder,
        cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS,
        cfg.TEST.DETECTIONS_THRESHOLD,
        cfg.TEST.DETECTIONS_PER_IMG,
        cfg.TEST.PRED_2D,
    )

    return postprocessor
