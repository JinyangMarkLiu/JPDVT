"""
Train a diffusion model on images.
"""
# import torch.distributed as dist
import numpy as np
import os
import gc

import torch

torch.cuda.empty_cache()
gc.collect()

import argparse
import sys

sys.path.insert(1, os.getcwd())
# sys.path.insert(1, '/diffusion_openai')
import dist_util, logger
# from diffusion_openai import dist_util, logger
from video_datasets import load_data
from resample import create_named_schedule_sampler
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from train_util import TrainLoop


def main():

    parser, defaults = create_argparser()
    args = parser.parse_args()
    parameters = args_to_dict(args, defaults.keys())
    # th.manual_seed(args.seed)
    # np.random.seed(args.seed)

    dist_util.setup_dist()
    logger.configure()
    for key, item in parameters.items():
        logger.logkv(key, item)
    logger.dumpkvs()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
   
    if len(args.model_path)>0:
        print("load ",args.model_path," to the model.")
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )

    model.to(dist_util.dev())
    # model.summary()
    # breakpoint()
    # import torch
    # import torchviz

    # model = ...  # create or load your PyTorch model


    print("device ",dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # print('model arch',model)
    # Save the model architecture to a file
    # with open("./figures/model_architecture.txt", "w") as f:
    #     f.write(str(model))

    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        rgb=args.rgb,
        seq_len=args.seq_len
    )

    # item=next(data)
    # breakpoint()

    if args.mask_range is None:
        mask_range = [0, args.seq_len]
    else:
        mask_range = [int(i) for i in args.mask_range if i != ","]
    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip=args.clip,
        anneal_type=args.anneal_type,
        steps_drop=args.steps_drop,
        drop=args.drop,
        decay=args.decay,
        max_num_mask_frames=args.max_num_mask_frames,
        mask_range=mask_range,
        uncondition_rate=args.uncondition_rate,
        exclude_conditional=args.exclude_conditional,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,  # -4
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,  # 8 for something
        microbatch=16,  # 32,  # -1 disables microbatches 32
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=32,  # 10
        save_interval=10000,  # 2000 100
        resume_checkpoint="",
        model_path="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip=1,
        seed=123,
        anneal_type=None,
        steps_drop=0.0,
        drop=0.0,
        decay=0.0,
        seq_len=32,  # 20
        max_num_mask_frames=6,  # 4
        mask_range=None,
        uncondition_rate=0,
        exclude_conditional=True,
        model_name='two_distribution',
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, defaults

if __name__ == "__main__":
    main()
    import json
    import numpy as np
