# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from newmodels import DiT_models
import argparse
from torch.utils.data import DataLoader
from newmodels import get_2d_sincos_pos_embed
from met import MET 
import numpy as np

from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    template = np.zeros((6,6))

    for i in range(6):
        for j in range(6):
            template[i,j] = 18 * i + j

    template = np.concatenate((template,template,template),axis=0)
    template = np.concatenate((template,template,template),axis=1)

    # Load model:
    # latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=args.image_size,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    print("Load model from:", args.ckpt )
    ckpt_path = args.ckpt 
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.train()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    split = "test"
    print("SPLIT:",split)
    dataset = MET('/data/mark/met3',split)


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True
    )

    time_emb = torch.tensor(get_2d_sincos_pos_embed(8, 3)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 18)).unsqueeze(0).float().to(device)
    time_emb_noise = torch.randn_like(time_emb_noise)
    time_emb_noise = time_emb_noise.repeat(1,1,1)

    model_kwargs = None

    def find_permutation(distance_matrix):
            sort_list = []
            for m in range(distance_matrix.shape[1]):
                order = distance_matrix[:,0].argmin()
                sort_list.append(order)
                distance_matrix = distance_matrix[:,1:]
                distance_matrix[order,:] = 10**5
            return sort_list

    abs_results = []
    for x in loader:
        model.load_state_dict(state_dict)
        x = x.to(device)
        indices = np.random.permutation(9)


        x = rearrange(x, 'b c (p1 h1) (p2 w1)-> b c (p1 p2) h1 w1',p1=3,p2=3,h1=96,w1=96)
        # breakpoint()
        x = x[:,:,indices,:,:]
        x = rearrange(x, ' b c (p1 p2) h1 w1->b c (p1 h1) (p2 w1)',p1=3,p2=3,h1=96,w1=96)
    # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward, x, time_emb_noise.shape, time_emb_noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        for sample,img in zip(samples,x):
            sample = rearrange(sample, '(p1 h1 p2 w1) d-> (p1 p2) (h1 w1) d',p1=3,p2=3,h1=6,w1=6)
            sample = sample.mean(1)
            dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
            order = find_permutation(dist)
            # order = np.array(order).reshape(3,3)
            pred = np.asarray(order).argsort()
            # breakpoint()
            print(pred)
            print(indices)
            abs_results.append(int((pred == indices).all()))
            if int((pred == indices).all()) == 0:
                img = rearrange(img, 'c (p1 h1) (p2 w1)-> c (p1 p2) h1 w1',p1=3,p2=3,h1=96,w1=96)
                img = img[:,pred.argsort()]
                img = rearrange(img, 'c (p1 p2) h1 w1->c (p1 h1) (p2 w1)',p1=3,p2=3,h1=96,w1=96)
                plt.imsave('image/'+str(len(abs_results))+'.png',img.permute(1,2,0).cpu().numpy()*0.5+0.5)


        print("test result on ",len(abs_results), "samples is :", np.asarray(abs_results).sum()/len(abs_results))

        if len(abs_results)>=2000:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="SJDiT")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512,288], default=288)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='/home/mark/projects/SJDiT_met3/results/001-SJDiT/checkpoints/0220000.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
