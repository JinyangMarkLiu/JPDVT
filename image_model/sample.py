"""
Solve Jigsaw Puzzles with JPDVT
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
from torch.utils.data import DataLoader
from models import get_2d_sincos_pos_embed
from datasets import MET 
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    template = np.zeros((6,6))

    for i in range(6):
        for j in range(6):
            template[i,j] = 18 * i + j

    template = np.concatenate((template,template,template),axis=0)
    template = np.concatenate((template,template,template),axis=1)

    # Load model:
    model = DiT_models[args.model](
        input_size=args.image_size,
    ).to(device)
    print("Load model from:", args.ckpt )
    ckpt_path = args.ckpt 
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    # Because the batchnorm doesn't work normally when batch size is 1
    # Thus we set the model to train mode
    model.train() 

    diffusion = create_diffusion(str(args.num_sampling_steps))
    if args.dataset == "met":
        # MET dataloader give out cropped and stitched back images
        dataset = MET(args.data_path,'test')
    elif args.dataset == "imagenet":
        dataset = ImageFolder(args.data_path, transform=transform)
    
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

    # find the order with a greedy algorithm
    def find_permutation(distance_matrix):
            sort_list = []
            for m in range(distance_matrix.shape[1]):
                order = distance_matrix[:,0].argmin()
                sort_list.append(order)
                distance_matrix = distance_matrix[:,1:]
                distance_matrix[order,:] = 2024
            return sort_list

    abs_results = []
    for x in loader:
        if args.dataset == 'imagenet':
            x, _ = x
        x = x.to(device)
        if args.dataset == 'imagenet' and args.crop:
            centercrop = transforms.CenterCrop((64,64))
            patchs = rearrange(x, 'b c (p1 h1) (p2 w1)-> b c (p1 p2) h1 w1',p1=3,p2=3,h1=96,w1=96)
            patchs = centercrop(patchs)
            x = rearrange(patchs, 'b c (p1 p2) h1 w1-> b c (p1 h1) (p2 w1)',p1=3,p2=3,h1=64,w1=64)

        # Generate the Puzzles
        indices = np.random.permutation(9)
        x = rearrange(x, 'b c (p1 h1) (p2 w1)-> b c (p1 p2) h1 w1',p1=3,p2=3,h1=args.image_size//3,w1=args.image_size//3)
        x = x[:,:,indices,:,:]
        x = rearrange(x, ' b c (p1 p2) h1 w1->b c (p1 h1) (p2 w1)',p1=3,p2=3,h1=args.image_size//3,w1=args.image_size//3)

        samples = diffusion.p_sample_loop(
            model.forward, x, time_emb_noise.shape, time_emb_noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        for sample,img in zip(samples,x):
            sample = rearrange(sample, '(p1 h1 p2 w1) d-> (p1 p2) (h1 w1) d',p1=3,p2=3,h1=args.image_size//48,w1=args.image_size//48)
            sample = sample.mean(1)
            dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
            order = find_permutation(dist)
            pred = np.asarray(order).argsort()
            abs_results.append(int((pred == indices).all()))
         
        print("test result on ",len(abs_results), "samples is :", np.asarray(abs_results).sum()/len(abs_results))

        if len(abs_results)>=2000 and args.dataset == "met":
            break
        if len(abs_results)>=50000 and args.dataset == "imagenet":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="JPDVT")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "met"], default="imagenet")
    parser.add_argument("--data-path", type=str,required=True)
    parser.add_argument("--crop", action='store_true', default=False)
    parser.add_argument("--image-size", type=int, choices=[192, 288], default=288)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args)
