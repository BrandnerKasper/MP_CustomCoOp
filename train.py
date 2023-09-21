import argparse

import torch

from dassl.utils import set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# Test
from general.arguments import Arguments
from general.custom_logger import setup_logger

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip


def generate_log_name(args: Arguments) -> str:
    name = ""
    if args.open_clip:
        name += "open_clip"
    else:
        name += "clip"
    name += "_" + args.backbone + "_" + args.pretrained + "_" + str(args.shots)
    return name


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    # if args.backbone:
    #     cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # Custom
    cfg.OPEN_CLIP = args.open_clip
        # set_open_clip()

    if args.pretrained:
        cfg.PRETRAINED = pretrained


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # cfg.TRAINER.ZeroshotCLIP.PREC = "fp32"

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple parser for the path to the dataset, "
                                                 "which backbone to use and if you like to use "
                                                 "CLIP or Open CLIP.")

    parser.add_argument("path", type=str, help='The path to the dataset in relation to this script')
    parser.add_argument("backbone", type=str, help='Which backbone to use, e.g. ResNet 50 or '
                                                   'ViT-B-32')
    parser.add_argument("--open_Clip", '-o', action="store_true", default=False,
                        help='Set to true if you like to use the Open Clip, else use Clip')
    parser.add_argument("--pretrained", '-p', type=str, default='openai',
                        help='The same backbones (Vit-B-32) are pretrained on different datasets. '
                             'This is only the case for open clip backbones')

    # Parse the command-line arguments
    parser_arg = parser.parse_args()

    # Access the parsed path string
    path_to_data = parser_arg.path  # /home/brandnerkasper/Uni/MP/MP_CustomCoOp/data
    backbone = parser_arg.backbone  # rn101, rn50, vit_b32, vit_b16, xlm-roberta-base-ViT-B-32
    open_Clip = parser_arg.open_Clip  # true for open_clip, false for clip
    pretrained = parser_arg.pretrained  # openai, laion5b_s13b_b90k, laion2b_s12b_b32k

    # print(f"Path to dataset {path_to_data}, backbone {backbone}, openclip {open_Clip}")

    # For the moment we only support CoOp and the caltech101 dataset
    args = Arguments("CoOp", path_to_data, "caltech101", backbone, "end", 16, 1, False,
                     "output/Caltech", open_Clip, pretrained)
    print(f"arguments: {args}")

    setup_logger(args.output_dir, generate_log_name(args))

    for i in 1, 2, 3:
        args.seed = i
        args.output_dir = args.output_dir + "/" + str(i)
        main(args)
