"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from matplotlib import gridspec
from skimage.metrics import structural_similarity, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from positional_encodings.torch_encodings import PositionalEncoding1D
sys.path.insert(1, os.getcwd())
import random

from video_datasets import load_data
import dist_util, logger
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from functools import reduce

import torch

from sklearn.metrics import pairwise_distances

def save_tensor(file_name, data_tensor):
    directory = "/home/wondmgezahu/ppo/latent-diffusion-main/EncodedLatent"
    file_path = os.path.join(directory, file_name)
    torch.save(data_tensor, file_path)
def plot_samples(generated_data, shuffled_input_data, ground_truth_data):
    # Delete the "figures" directory and its contents (if it exists)
    if os.path.exists('fig_test'):
        shutil.rmtree('fig_test')

    # Create a new "figures" directory
    os.makedirs('fig_test')

    for i in range(generated_data.shape[0]):  #
        # Select the current batch
        batch_generated = generated_data[i, :, :, :, :]
        batch_shuffled = shuffled_input_data[i, :, :, :, :]
        batch_ground_truth = ground_truth_data[i, :, :, :, :]

        # Create a figure with 10 subplots (one for each frame)
        fig = plt.figure(figsize=(6, 6))  # (3,3) for 4 frame
        gs = gridspec.GridSpec(nrows=3, ncols=generated_data.shape[2], left=0, right=1, top=1, bottom=0)

        # Reduce the space between the subplots
        # fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        # Loop over the frames
        for j in range(batch_generated.shape[1]):
            # Select the current frame
            frame_generated = batch_generated[:, j, :, :]
            frame_shuffled = batch_shuffled[:, j, :, :]
            frame_ground_truth = batch_ground_truth[:, j, :, :]

            # Create a subplot at the current position
            ax1 = fig.add_subplot(gs[0, j])
            ax2 = fig.add_subplot(gs[1, j])
            ax3 = fig.add_subplot(gs[2, j])

            # Plot the frame in grayscale
            # breakpoint()
            ax1.imshow(np.squeeze(frame_shuffled), cmap='gray')  # cmap='gray'
            ax1.axis('off')
            if j == 4:
                ax1.set_title('shuffled samples', loc='center', fontsize=10)

            ax2.imshow(np.squeeze(frame_ground_truth), cmap='gray')  # cmap='gray'
            ax2.axis('off')
            if j == 4:
                ax2.set_title('ground truth samples', loc='center', fontsize=10)

            ax3.imshow(np.squeeze(frame_generated), cmap='gray')  # cmap='gray'
            ax3.axis('off')
            if j == 4:
                ax3.set_title('generated samples', loc='center', fontsize=10)

        plt.savefig('fig_test/batch{}.png'.format(i))

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    # logger.configure()
    if args.seed:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    # breakpoint()
    model.train()

    cond_kwargs = {}
    cond_frames = []
    if args.cond_generation:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            # rgb=args.rgb,
            seq_len=args.seq_len
        )

        num = ""

        for i in args.cond_frames:
            if i == ",":
                cond_frames.append(int(num))
                num = ""
            else:
                num = num + i
        ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
        # logger.log(f"cond_frames: {cond_frames}")
        # logger.log(f"ref_frames: {ref_frames}")
        # logger.log(f"seq_len: {args.seq_len}")
        cond_kwargs["resampling_steps"] = args.resample_steps
    cond_kwargs["cond_frames"] = cond_frames

    channels = 4
    # breakpoint()
    # logger.log("sampling...")
    all_videos = []
    all_gt = []
    shuffled_input = []
    ground_truth = []
    sample_time=[]
    perm_index = []
    imputation_index = []
    all_idx = []
    all_permutation = []
    all_names = []
    all_sampled_normalized=[]
    mask_list=[]
    kendall_dis_sum = 0
    kendall_dis_list = []
    while len(kendall_dis_list) < args.num_samples:
        print(len(kendall_dis_list))
        min_val, max_val = 0, 0
        if args.cond_generation:
            raw_video, _,index = next(data)  # video, _ = next(data) for others except something and nturgbd-skeleton dataset
            # breakpoint()
            raw_video = raw_video.permute(0,2,1,3,4)
            # all_names.append(name)
            # raw_video = raw_video.permute(0, 2, 1, 3, 4)  # permuted for something dataset and nturgbd-skeleton only
            # original_skeleton = video.clone().detach()
            video = raw_video.float()  # just for skeleton and nturgbd dataset
            # breakpoint()
            # normalization for nturgbd
            min_val = torch.min(video)
            max_val = torch.max(video)
            range_val = max_val - min_val

            batch = (video - min_val) / range_val
            video = batch * 2 - 1
            # cond_kwargs["cond_img"] = video[:, :, cond_frames].to(dist_util.dev())
            # cond_kwargs["cond_frames"] = cond_frames
            # video = video.to(dist_util.dev())
            # breakpoint()
        video = video
        idx = th.randperm(video.shape[2])
        idx=idx[idx!=0]
        idx=torch.cat((torch.tensor([0]),idx))
        # breakpoint()
        perm_index.append(idx)
        # original_data = torch.Tensor(video)
        
        conditional_data = video[:, :, idx, :, :]
        true_idx = np.zeros(args.seq_len, )
        true_idx[:] = idx

        # Remove some frames
        num_frames_to_remove = 12
        print("number of missing frame is: ",num_frames_to_remove)
        #
        # # create an array of indices for all frames
        frame_indices = np.arange(conditional_data.shape[2])
        #
        # # randomly select the indices of the frames to remove for all batches
        remove_indices = np.random.choice(frame_indices, num_frames_to_remove, replace=False)
        remove_indices = remove_indices[remove_indices!=0]
        #
        # # create an array of zeros with the same shape as the video frames
        dummy_frames = np.random.randn(
            conditional_data.shape[0], conditional_data.shape[1], 1, conditional_data.shape[3],
             conditional_data.shape[4])
        dummy_frames = torch.from_numpy(dummy_frames).to(conditional_data.device)
        # breakpoint()
        conditional_data[:, :, remove_indices, :, :] = dummy_frames.float()
        
        mask = np.ones(conditional_data.shape[2])
        mask[remove_indices] = 0
        mask_list.append(mask)
        mask = torch.tensor(mask).cuda()

        
        cond_kwargs["cond_frames"] = [i for i in frame_indices if i not in remove_indices]
        cond_kwargs["cond_img"] = conditional_data[:, :, cond_kwargs["cond_frames"]].to(dist_util.dev())
        # breakpoint()
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.seq_len, 16),
            clip_denoised=args.clip_denoised,
            progress=False,
            cond_kwargs=cond_kwargs,
            condition_args=conditional_data,
            # mask=mask,
            original_args=video
            # gap=args.diffusion_step_gap
        )
        # breakpoint()
        p_enc_1d_model = PositionalEncoding1D(16)
        time_emb_gt = p_enc_1d_model(torch.rand(sample[0].shape[0], sample[0].shape[1], 16))
        

        def find_permutation(distance_matrix):
            sort_list = []
            for m in range(distance_matrix.shape[1]):
                order = distance_matrix[:,0].argmin()
                sort_list.append(order)
                distance_matrix = distance_matrix[:,1:]
                distance_matrix[order,:] = 10**5
            return sort_list
                # conditional_data = ((conditional_data.clamp(-1,1) + 1) * (max_val-min_val) /2+ min_val)
        # sample = sample.contiguous()
        sample_normalized = sample[0].clamp(-1, 1)
        dist = pairwise_distances(time_emb_gt[0], sample[0][0].cpu(), metric='manhattan')
        permutation = find_permutation(dist)
        permutation = np.array(permutation)
        idx = np.array(idx)

        if num_frames_to_remove != 0:
            # breakpoint()
            permutation_con = permutation[cond_kwargs["cond_frames"]]
            idx_con = idx[cond_kwargs["cond_frames"]]
            permutation_blank = np.sort(permutation[remove_indices])
            idx_blank = np.sort(idx[remove_indices])

            permutation = np.concatenate((permutation_con,permutation_blank))
            idx = np.concatenate((idx_con,idx_blank))
            

        kendall_dis_1 = 0
        for n1 in range(len(permutation)):
            for n2 in range(len(permutation)):
                if permutation[n1]<permutation[n2] and idx[n1]>idx[n2]:
                    kendall_dis_1 += 1
        # kendall_dis_1 +=  np.abs(permutation - idx).sum()
        kendall_dis_2 = 0
        for n1 in range(len(permutation)):
            for n2 in range(len(permutation)):
                if permutation[n1] < permutation[n2] and (args.seq_len-1-idx)[n1] > (args.seq_len-1-idx)[n2]:
                    kendall_dis_2 += 1
        # kendall_dis_2 +=  np.abs(permutation -args.seq_len + 1 + idx).sum()
        kendall_dis =min(kendall_dis_1,kendall_dis_2)/(sample[0].shape[1]*(sample[0].shape[1]-1)/2)

        
        # breakpoint()
        
        kendall_dis_sum += kendall_dis
        kendall_dis_list.append(kendall_dis)
        print(len(kendall_dis_list), 'kendall distance:', kendall_dis,kendall_dis_sum/len(kendall_dis_list)  )
        if kendall_dis>0.05:
            print('video_index:',index)
            print(idx)
            print(permutation)
       

    print("Total result is:",kendall_dis_sum/args.num_samples)
        # # end of normalization
        # all_sampled_normalized.append(sample_normalized)
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # # gathered_samples_time=[th.zeros_like(sample_time_out) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_videos.extend([sample.cpu().numpy() for sample in gathered_samples])
        # # sample_time.append(sample_time_out.cpu().numpy()) #for sample_time_out in gathered_samples_time])
        # ground_truth.append(video.permute(0, 2, 3, 4, 1).cpu().numpy())
        # shuffled_input.append(conditional_data.permute(0, 2, 3, 4, 1).cpu().numpy())
        # all_idx.append(true_idx)
        # # breakpoint()
        # # logger.log(f"created {len(all_videos) * args.batch_size} samples")

    # generated_data = np.concatenate(all_videos, axis=0)
    # ground_truth_data = np.asarray(ground_truth)
    # shuffled_input_data = np.asarray(shuffled_input)

    # gen_raw = np.concatenate(all_videos, axis=0)  # raw BxTxWxHXC 1x20x32x32x4
    # gen_raw = torch.from_numpy(gen_raw).float()
    # gen_norm = torch.cat(all_sampled_normalized, dim=0)
    # gen_time=np.concatenate(sample_time,axis=0)
    # breakpoint()
    # plot_samples(generated_data, shuffled_input_data, ground_truth_data)
    # fig=plt.figure()
    # for i in range(gen_time.shape[0]):
    #     plt.imshow(gen_time[i])
    #     plt.savefig('fig_test/time{}.png'.format(i))

    # # breakpoint()
    # p_enc_1d_model = PositionalEncoding1D(16)
    # time_emb = p_enc_1d_model(torch.rand(generated_data.shape[0], generated_data.shape[1], 16))
    # # distances = torch.cdist(torch.tensor(generated_data[0]), torch.tensor(time_emb[0]))
    # # indices = distances.argmin(dim=-1)
    # indice_list=[]
    # for i in range(generated_data.shape[0]):
    #     distances = torch.cdist(torch.tensor(generated_data[i]), torch.tensor(time_emb[0]))
    #     indices = distances.argmin(dim=-1)
    #     indice_list.append(indices)
    # # breakpoint()
    # from datetime import datetime
    # currentDateAndTime = datetime.now()
    # if args.cond_generation and args.save_gt:
    #     np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' +str(datetime.now().day)+'_'+str(datetime.now().hour)+ '_gt_'+str(args.resample_steps)+str(args.diffusion_step_gap)+'.npy'),ground_truth_data)
    # np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' +str(datetime.now().day)+'_'+str(datetime.now().hour)+ '_gen_'+str(args.resample_steps)+str(args.diffusion_step_gap)+'.npy'),
    #         generated_data)
    # np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' +str(datetime.now().day)+'_'+str(datetime.now().hour)+ '_idx_'+str(args.resample_steps)+str(args.diffusion_step_gap)+'.npy'),
    #         np.asarray(all_idx))
    # np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' + str(datetime.now().day) + '_' + str(
    #     datetime.now().hour) + '_file_names_' + str(args.resample_steps) + str(args.diffusion_step_gap) + '.npy'),
    #         np.asarray(all_names))
    # np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' +str(datetime.now().day)+'_'+str(datetime.now().hour)+ '_condition_'+str(args.resample_steps)+str(args.diffusion_step_gap)+'.npy'),
    #         shuffled_input_data)
    # np.save(os.path.join('GeneratedLatent', str(args.num_samples) + '_samples_' +str(datetime.now().day)+'_'+str(datetime.now().hour)+ '_maskList_'+str(args.resample_steps)+str(args.diffusion_step_gap)+'.npy'),
    #         np.asarray(mask_list))
    # dist.barrier()
    # logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,  # 10
        batch_size=1,  # 10
        use_ddim=False,
        model_path="",
        seq_len=32,  # 16
        sampling_type="generation",
        cond_frames="",
        cond_generation=True,  # True
        resample_steps=1,
        data_dir='',
        save_gt=True,
        seed=0,
        diffusion_step_gap=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import time

    MODEL_FLAGS = "--image_size 8 --num_channels 128 --num_res_blocks 3 --scale_time_dim 8"  # image size 64
    DIFFUSION_FLAGS = "--diffusion_steps 1200 --noise_schedule linear"  # 1000
    start = time.time()
    main()
    end = time.time()
    print(f"elapsed time: {end - start}")