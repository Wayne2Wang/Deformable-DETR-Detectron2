# Modified by Zilin Wang from 
# https://github.com/facebookresearch/detr/blob/main/d2/detr/detr.py
# https://github.com/fundamentalvision/Deformable-DETR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.structures import Instances

from deformable_detr.models.backbone import Joiner, Backbone
from deformable_detr.models.deformable_detr import DeformableDETR as DeformableDETRModel, SetCriterion
from deformable_detr.models.matcher import HungarianMatcher
from deformable_detr.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from deformable_detr.models.deformable_transformer import DeformableTransformer
from deformable_detr.models.segmentation import DETRsegm, PostProcessSegm
from deformable_detr.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from deformable_detr.util.segm_ops import convert_coco_poly_to_mask

__all__ = ["DeformableDETR"]


@META_ARCH_REGISTRY.register()
class DeformableDETR(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # read from config
        args = cfg.MODEL.DEFORMABLE_DETR
        self.device = cfg.MODEL.DEVICE
        self.masks = cfg.MODEL.MASK_ON
        self.top_k_pred = cfg.TEST.DETECTIONS_PER_IMAGE

        # build position embedding
        N_steps = args.TRANSFORMER.HIDDEN_DIM // 2
        if args.POSITION_EMBEDDING in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif args.POSITION_EMBEDDING in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {args.POSITION_EMBEDDING}")
        
        # build backbone
        train_backbone = (cfg.SOLVER.BACKBONE_LR_MULTIPLIER * cfg.SOLVER.BASE_LR) > 0
        return_interm_layers = self.masks or (args.NUM_FEATURE_LEVELS > 1)
        backbone = Backbone(args.BACKBONE, train_backbone, return_interm_layers, args.DILATION)
        backbone = Joiner(backbone, position_embedding)

        # transformer
        transformer = DeformableTransformer(
            d_model=args.TRANSFORMER.HIDDEN_DIM,
            nhead=args.TRANSFORMER.NHEADS,
            num_encoder_layers=args.TRANSFORMER.ENC_LAYERS,
            num_decoder_layers=args.TRANSFORMER.DEC_LAYERS,
            dim_feedforward=args.TRANSFORMER.DIM_FEEDFORWARD,
            dropout=args.TRANSFORMER.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=args.NUM_FEATURE_LEVELS,
            dec_n_points=args.TRANSFORMER.DEC_N_POINTS,
            enc_n_points=args.TRANSFORMER.ENC_N_POINTS,
            two_stage=args.TWO_STAGE,
            two_stage_num_proposals=args.TRANSFORMER.NUM_QUERIES
        )

        self.deformable_detr = DeformableDETRModel(
            backbone,
            transformer,
            num_classes=args.NUM_CLASSES,
            num_queries=args.TRANSFORMER.NUM_QUERIES,
            num_feature_levels=args.NUM_FEATURE_LEVELS,
            aux_loss=args.LOSS.AUX_LOSS,
            with_box_refine=args.WITH_BOX_REFINE,
            two_stage=args.TWO_STAGE,
        )
        # segmentation head
        if self.masks:
            if args.FROZEN_WEIGHTS != '':
                print("LOAD pre-trained weights")
                weight = torch.load(args.FROZEN_WEIGHTS, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k: # TODO: fix this
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.deformable_detr.load_state_dict(new_weight)
                del new_weight
            self.deformable_detr = DETRsegm(self.deformable_detr, freeze_detr=(args.FROZEN_WEIGHTS is not None))
        self.deformable_detr.to(self.device)

        # matchers
        matcher = HungarianMatcher(
            cost_class=args.MATCHER.SET_COST_CLASS,
            cost_bbox=args.MATCHER.SET_COST_BBOX,
            cost_giou=args.MATCHER.SET_COST_GIOU
        )

        # losses
        weight_dict = {'loss_ce': args.LOSS.CLS_LOSS_COEF, 'loss_bbox': args.LOSS.BBOX_LOSS_COEF}
        weight_dict['loss_giou'] = args.LOSS.GIOU_LOSS_COEF
        if self.masks:
            weight_dict["loss_mask"] = args.LOSS.MASK_LOSS_COEF
            weight_dict["loss_dice"] = args.LOSS.DICE_LOSS_COEF
        # TODO this is a hack
        if args.LOSS.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(args.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['labels', 'boxes', 'cardinality']
        if self.masks:
            losses += ["masks"]
        # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
        self.criterion = SetCriterion(args.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=args.LOSS.FOCAL_ALPHA)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.deformable_detr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.masks else None
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.masks and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    # Similar to the PostProcess() in the original codebase but without the top k filtering
    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (cx,cy,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # deformable detr uses a different scheme for classifying boxes
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), self.top_k_pred, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, boxes, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.masks:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
