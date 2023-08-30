# Modified by Zilin Wang from
# https://github.com/facebookresearch/detr/blob/main/d2/train_net.py
# https://github1s.com/fundamentalvision/Deformable-DETR/blob/main/main.py

"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

# Note: if multi-gpu hangs set NCCL_P2P_DISABLE=1

import os
import itertools
from typing import Any, Dict, List, Set

import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.engine.defaults import DefaultTrainer

from modeling import DeformableDETR
from deformable_detr.util.dataset_mapper import DetrDatasetMapper
from config import add_deformable_detr_config



class Trainer(DefaultTrainer):
    """
    Extension of the Training class adapted to Deformable-DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.META_ARCHITECTURE == "DeformableDETR":
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.META_ARCHITECTURE == "DeformableDETR":
            mapper = DetrDatasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.META_ARCHITECTURE == "DeformableDETR":
            def match_name_keywords(n, name_keywords):
                out = False
                for b in name_keywords:
                    if b in n:
                        out = True
                        break
                return out
            params = [
                {
                    "params":
                        [p for n, p in model.named_parameters()
                        if not match_name_keywords(n, cfg.SOLVER.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.SOLVER.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                    "lr": cfg.SOLVER.BASE_LR,
                },
                {
                    "params": [p for n, p in model.named_parameters() if match_name_keywords(n, cfg.SOLVER.LR_BACKBONE_NAMES) and p.requires_grad],
                    "lr": cfg.SOLVER.BASE_LR*cfg.SOLVER.BACKBONE_LR_MULTIPLIER,
                },
                {
                    "params": [p for n, p in model.named_parameters() if match_name_keywords(n, cfg.SOLVER.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                    "lr": cfg.SOLVER.BASE_LR*cfg.SOLVER.BACKBONE_LR_MULTIPLIER,
                }
            ]
        else:
            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for key, value in model.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "backbone" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_LR_MULTIPLIER
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    if 'deformable_detr' in args.config_file:
        add_deformable_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )