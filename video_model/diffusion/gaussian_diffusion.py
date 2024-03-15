# """
# This code started out as a PyTorch port of Ho et al's diffusion models:
# https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
#
# Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
# """

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import pdb
import shutil

import math
import random

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os

from matplotlib import gridspec

from nn import mean_flat, sum_flat
from losses import normal_kl, discretized_gaussian_log_likelihood
import torch.nn.functional as F
import torch

from positional_encodings.torch_encodings import PositionalEncoding1D
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,

    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # for harmonizing
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        # breakpoint()
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None, condition_args=None,
            original_args=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        # breakpoint()
        video_output,model_output = model(condition_args.to(x.device), self._scale_timesteps(t).to(x.device),
                                              x.to(x.device))  # model output，,recons_time_emb , condition_args.cuda()
        # print('recns loss = ', ((recons_video-video.cuda())**2).mean())
        # plt.imshow(model_output[0].cpu().detach().numpy());plt.savefig('plots_sample/' + str(t[0].cpu().numpy()) + '_out.png')
        # x=condition_args.cuda()
        # breakpoint()
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            model_output, model_var_values = th.split(model_output, C, dim=1)
            video_outptu,video_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                video_log_variance = video_var_values
                model_variance = th.exp(model_log_variance)
                video_variance = th.exp(video_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                video_min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, condition_args.shape
                )
                video_max_log = _extract_into_tensor(np.log(self.betas), t, condition_args.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                video_frac = (video_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
                vidoe_log_variance = video_frac * video_max_log + (1 - video_frac) * video_min_log
                video_variance = th.exp(video_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance_ = model_variance.copy()
            model_log_variance_ = model_log_variance.copy()
            model_variance = _extract_into_tensor(model_variance_, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance_, t, x.shape)
            video_variance = _extract_into_tensor(model_variance_, t, condition_args.shape)
            video_log_variance = _extract_into_tensor(model_log_variance_, t, condition_args.shape)

        def process_xstart(x):
            # breakpoint()
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # if self.model_mean_type == ModelMeanType.PREVIOUS_X:
        #     pred_xstart = process_xstart(
        #         self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        #     )
        #     model_mean = model_output
        # breakpoint()
        model_mean_type = ModelMeanType.EPSILON
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # breakpoint()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        video_mean_type = ModelMeanType.EPSILON
        if video_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if video_mean_type == ModelMeanType.START_X:
                video_pred_xstart = process_xstart(video_output)
            else:
                video_pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=condition_args, t=t, eps=video_output)
                )
            # breakpoint()
            video_mean, _, _ = self.q_posterior_mean_variance(
                x_start=video_pred_xstart, x_t=condition_args, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        assert (
                video_mean.shape == video_log_variance.shape == video_pred_xstart.shape == condition_args.shape
        )
        return {
            "mean": model_mean,
            "video_mean": video_mean,
            "variance": model_variance,
            "video_variance":video_variance,
            "log_variance": model_log_variance,
            "video_log_variance": video_log_variance,
            "pred_xstart": pred_xstart,
            "video_pred_xstart":video_pred_xstart
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return (t.float() * (1000.0 / self.num_timesteps)).int()
        return t

    def p_sample(
            self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None, condition_args=None,
            original_args=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            condition_args=condition_args,
            original_args=original_args

        )
        noise = th.randn_like(x)
        video_noise = th.randn_like(condition_args)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        video_nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(condition_args.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        video_sample = out["video_mean"] + video_nonzero_mask * th.exp(0.5 * out["video_log_variance"]) * video_noise
        # sample = out["mean"] + th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"video_sample": video_sample, "video_pred_xstart": out["video_pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            cond_kwargs=None,
            condition_args=None,
            original_args=None,
            gap=1,
            recons_frame=None,
            conditional_frame=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None

        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                cond_kwargs=cond_kwargs,
                condition_args=condition_args,
                original_args=original_args,
                gap=gap,
                recons_frame=recons_frame,
                conditional_frame=conditional_frame
        ):
            final = sample
        return final['sample'],final['video_sample']

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            cond_kwargs=None,
            condition_args=None,
            original_args=None,
            gap=1,
            recons_frame=None,
            conditional_frame=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise.to(device)
        else:
            img = th.randn(*shape, device=device)
        condition = condition_args.to(device)
        p_enc_1d_model = PositionalEncoding1D(16)
        time_emb = p_enc_1d_model(th.rand(condition.shape[0], condition.shape[2], 16)).to(device)
        indices = list(range(self.num_timesteps))[::-1]
        time_emb=th.randn_like(time_emb)
        # breakpoint()

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        resampling_steps = 1
        cond_frames = cond_kwargs["cond_frames"]
        cond_img = cond_kwargs["cond_img"]
        if cond_frames:
            cond_img = cond_kwargs["cond_img"]
            resampling_steps = cond_kwargs["resampling_steps"]

        # directory for the plots
        if os.path.exists('figures_samples'):
            shutil.rmtree('figures_samples')
        img=condition.to(device)
        # Create a new "figures" directory
        # os.makedirs('figures_samples')
        # breakpoint()
        for i in indices[::gap]:
            t = th.tensor([i] * shape[0], device=device)

            for r in range(resampling_steps):
                # img[:, :, conditional_frame] = condition[:, :, conditional_frame]
                # breakpoint()
                if cond_frames:
                    img[:,:,cond_frames] = cond_img
                    time_emb[:,0,:] = p_enc_1d_model(time_emb)[:,0,:] 
                    # breakpoint()
                with th.no_grad():
                    out = self.p_sample(
                        model,
                        time_emb,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        condition_args=img,
                        original_args=original_args
                    )
                # 
                time_emb = out["sample"]
                img=out["video_sample"]
                # breakpoint()
                # x_start_pred = out["pred_xstart"]
                if r < resampling_steps - 1:  #
                    time_emb = self.forward_diffusion(time_emb, device, i)
            
            yield out


    def forward_diffusion(
            self,
            x_start,
            device,
            i,
    ):
        """ starting from x_i, create a sample after length forward diffusion steps
      i.e. x_t+1 ~ q(x_t+1|x_t)

        Args:
            x_start: the tensor at time x_i
            device: gpu or cpu device on which inference is done
            i: starting time of the diffusion process
            length: amount of diffusion steps

        Returns:
            tensor sample x_t+1
        """

        noise = th.randn_like(x_start, device=device)
        # breakpoint()
        t = th.tensor([i] * x_start.shape[0], device=device)
        x_t = (_extract_into_tensor(self.sqrt_alphas, t, x_start.shape) * x_start
               + _extract_into_tensor(self.sqrt_betas, t, x_start.shape)
               * noise)

        return x_t

    def resampling(
            self,
            x_start,
            model,
            clip_denoised,
            denoised_fn,
            model_kwargs,
            device,
            t,
            jump_length,
            use_ddim,
    ):

        sample_fn = self.p_sample if not use_ddim else self.ddim_sample
        indices = list(range(t, t + jump_length))

        x_t = x_start
        for i in indices:
            x_t = self.forward_diffusion(x_t, device, i)

        for i in indices[::-1]:
            t = th.tensor([i] * x_t.shape[0], device=device)
            with th.no_grad():
                out = sample_fn(
                    model,
                    x_t,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
            x_t = out['sample']
        return x_t

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            condition_args=None,
            original_args=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            condition_args=condition_args,
            original_args=original_args
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            cond_kwargs=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                cond_kwargs=cond_kwargs
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            cond_kwargs=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        saver = cond_kwargs["saver"]
        resampling_steps = 1
        cond_frames = cond_kwargs["cond_frames"]
        if cond_frames:
            cond_img = cond_kwargs["cond_img"]
            resampling_steps = cond_kwargs["resampling_steps"]

        p_enc_1d_model = PositionalEncoding1D(16)
        time_emb_gt = p_enc_1d_model(time_emb)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            for r in range(resampling_steps):
                if cond_frames:
                    img[:, :, cond_frames] = cond_img
                    time_emb[:,0,:] = time_emb_gt[:,0,:] 

                    with th.no_grad():
                        out = self.p_sample(
                            model,
                            time_emb,
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                            condition_args=img,
                            original_args=original_args
                        )

                    img = out["sample"]
                    time_emb = out["pred_xstart"]
                    # img=out["video_sample"]
                    print('Mean value at step:', t, 'is: ', image.mean())
                    if r < resampling_steps - 1:
                        img = self.forward_diffusion(img, device, i)

                yield out

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, masks=[]
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        for b, mask in enumerate(masks):
            kl[b, :, mask] = th.zeros_like(kl[b, :, mask])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _vb_terms_bpd_unshuffle(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        # for b, mask in enumerate(masks):
        #     kl[b, :, mask] = th.zeros_like(kl[b, :, mask])
        # kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    def training_losses(
            self,
            model,
            x_start,
            t,
            condition_data=None,
            model_kwargs=None,
            noise=None,
            max_num_mask_frames=2,
            mask_range=None,
            uncondition_rate=0,
            exclude_conditional=True
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        masks = th.ones_like(x_t)

        for b in range(x_t.shape[0]):
            unconditional = np.random.binomial(1, uncondition_rate)
            # breakpoint()
            if unconditional:
                mask = []
            else:
                # r = random.randint(1, max_num_mask_frames)
                # mask = random.sample(range(*mask_range), r)
                mask = 0
                # breakpoint()

            masks[b, :, mask] = 0

        noise *= masks
        x_t = x_t * masks + (1 - masks) * x_start
        terms = {}
        # breakpoint()

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            model_output = model(x_t, self._scale_timesteps(t), condition_data)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=True,
                    masks=masks
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            # breakpoint()
            assert model_output.shape == target.shape == x_start.shape

            mse = (target - model_output) ** 2
            if exclude_conditional:
                mse *= masks
                terms["mse"] = sum_flat(mse) / sum_flat(masks)
            else:
                terms["mse"] = mean_flat(mse)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def training_losses_two_dis(
            self,
            model,
            x_start,
            t,
            condition_data=None,
            time_emb=None,
            model_kwargs=None,
            noise=None,
            max_num_mask_frames=2,
            mask_range=None,
            uncondition_rate=0,
            exclude_conditional=True
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        video_condition = condition_data
        video_start = x_start
        time_emb_start = time_emb
        video_noise = th.randn_like(video_start)
        time_emb_noise = th.randn_like(time_emb_start)

        video_t = self.q_sample(video_start, t, noise=video_noise)
        time_emb_t = self.q_sample(time_emb_start, t, noise=time_emb_noise)
        # fix the first time emb
        time_emb_t[:,0,:] = time_emb_start[:,0,:]
        time_emb_noise[:,0,:] = 0
        # added
        masks = th.ones_like(video_t)
        masks_time_emb = th.ones_like(time_emb_start)
        condition_frames = []

        ###### SET # MISSING FRAMES HERE
        # 8-2
        r = np.random.randint(24,30)

        mask = random.sample(range(masks.shape[2]), r)
        # condition_frames.append(mask)
        masks[:, :, mask] = 0
        masks[:, :, 0] = 0
        masks_time_emb[:, mask, :] = 0
        masks_time_emb[:, 0, :] = 0
        # breakpoint()
        # condition_frames = torch.tensor(condition_frames).cuda()
        condition_frames = torch.tensor(mask)
        condition_frames = condition_frames.unsqueeze(0).repeat(masks.shape[0], 1)
        # breakpoint()
        video_noise *= masks
        # noise_time_emb *= masks_time_emb
        video_t = video_t * masks + (1 - masks) * video_start  # masked -- keep the frame from the diffusion

        terms = {}
        # breakpoint()

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            video_output, time_emb_output = model(video_t, self._scale_timesteps(t), time_emb_t)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:

                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target_video = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=video_start, x_t=video_t, t=t
                )[0],
                ModelMeanType.START_X: video_start,
                ModelMeanType.EPSILON: video_noise,
            }[self.model_mean_type]
            target_time_emb = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=time_emb_start, x_t=time_emb_t, t=t
                )[0],
                ModelMeanType.START_X: time_emb_start,
                ModelMeanType.EPSILON: time_emb_noise,
            }[self.model_mean_type]
            assert time_emb_output.shape == target_time_emb.shape == time_emb_start.shape

            # def contrastive_loss(time_emb_gen):
            #     batch_size, num_indices, _ = time_emb_gen.shape
            #     sim_matrix = F.mse_loss(time_emb_gen.unsqueeze(1), time_emb_gen.unsqueeze(1), reduction='none').mean(dim=-1)
            #     sim_matrix = sim_matrix.mean(dim=0)
            #     pos_sim = sim_matrix.diagonal().sum()
            #     neg_sim = sim_matrix.sum() - pos_sim
            #     loss = pos_sim - neg_sim
            #     return loss / (num_indices * num_indices)

            def contrastive_loss(condition_frames, time_emb_gen):
                # apply contrastive loss on available frames only

                mask_frame = torch.zeros(time_emb_gen.shape[0], time_emb_gen.shape[1], dtype=torch.bool)

                # Set the values at the indices specified in exclude_frames to False
                for i in range(condition_frames.shape[0]):
                    mask_frame[i][condition_frames[i]] = True
                # time_emb_gen = torch.stack([time_emb_gen[i, mask_frame[i], :] for i in range(time_emb_gen.shape[0])])
                # breakpoint()
                batch_size, num_indices, _ = time_emb_gen.shape
                sim_matrix = F.mse_loss(time_emb_gen.unsqueeze(1), time_emb_gen.unsqueeze(1), reduction='none').mean(
                    dim=-1)
                sim_matrix = sim_matrix.mean(dim=0)
                pos_sim = sim_matrix.diagonal().sum()
                neg_sim = sim_matrix.sum() - pos_sim
                loss = pos_sim - neg_sim
                return loss / (num_indices * num_indices)

            import torch.nn.functional as F

            def verification_loss(condition_frames, time_emb_gt, time_emb_gen):
                mask_frame = torch.zeros(time_emb_gen.shape[0], time_emb_gen.shape[1], dtype=torch.bool)

                # Set the values at the indices specified in exclude_frames to False
                for i in range(condition_frames.shape[0]):
                    mask_frame[i][condition_frames[i]] = True
                # time_emb_gen = torch.stack([time_emb_gen[i, mask_frame[i], :] for i in range(time_emb_gen.shape[0])])
                # time_emb_gt = torch.stack([time_emb_gt[i, mask_frame[i], :] for i in range(time_emb_gt.shape[0])])
                # breakpoint()
                column_sum_gt = time_emb_gt.sum(dim=1, keepdim=True)
                column_sum_gen = time_emb_gen.sum(dim=1, keepdim=True)

                loss = (column_sum_gen - column_sum_gt) ** 2

                return loss

                # mse loss

            mse_video = 0.2 * (video_noise - video_output) ** 2
            # mse_video *= masks
            mse_time_emb = 0.8 * (time_emb_noise - time_emb_output) ** 2
            # breakpoint()
            contra_loss = contrastive_loss(condition_frames, time_emb_output)
            verify_loss = verification_loss(condition_frames, time_emb_start, time_emb_output)

            terms["mse"] = mean_flat(mse_video)  + mean_flat(mse_time_emb) #+ 0.1 * contra_loss + 0.1 * mean_flat(
                # verify_loss)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None, progress=True):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for t in indices:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

#
# import enum
# import pdb
#
# import math
# import random
#
# import numpy as np
# import torch as th
# import matplotlib.pyplot as plt
# import os
#
# from nn import mean_flat, sum_flat
# from losses import normal_kl, discretized_gaussian_log_likelihood
#
#
# def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
#     """
#     Get a pre-defined beta schedule for the given name.
#
#     The beta schedule library consists of beta schedules which remain similar
#     in the limit of num_diffusion_timesteps.
#     Beta schedules may be added, but should not be removed or changed once
#     they are committed to maintain backwards compatibility.
#     """
#     if schedule_name == "linear":
#         # Linear schedule from Ho et al, extended to work for any number of
#         # diffusion steps.
#         scale = 1000 / num_diffusion_timesteps
#         beta_start = scale * 0.0001
#         beta_end = scale * 0.02
#         return np.linspace(
#             beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
#         )
#     elif schedule_name == "cosine":
#         return betas_for_alpha_bar(
#             num_diffusion_timesteps,
#             lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
#         )
#     else:
#         raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
#
#
# def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
#     """
#     Create a beta schedule that discretizes the given alpha_t_bar function,
#     which defines the cumulative product of (1-beta) over time from t = [0,1].
#
#     :param num_diffusion_timesteps: the number of betas to produce.
#     :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
#                       produces the cumulative product of (1-beta) up to that
#                       part of the diffusion process.
#     :param max_beta: the maximum beta to use; use values lower than 1 to
#                      prevent singularities.
#     """
#     betas = []
#     for i in range(num_diffusion_timesteps):
#         t1 = i / num_diffusion_timesteps
#         t2 = (i + 1) / num_diffusion_timesteps
#         betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
#     return np.array(betas)
#
#
# class ModelMeanType(enum.Enum):
#     """
#     Which type of output the model predicts.
#     """
#
#     PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
#     START_X = enum.auto()  # the model predicts x_0
#     EPSILON = enum.auto()  # the model predicts epsilon
#
#
# class ModelVarType(enum.Enum):
#     """
#     What is used as the model's output variance.
#
#     The LEARNED_RANGE option has been added to allow the model to predict
#     values between FIXED_SMALL and FIXED_LARGE, making its job easier.
#     """
#
#     LEARNED = enum.auto()
#     FIXED_SMALL = enum.auto()
#     FIXED_LARGE = enum.auto()
#     LEARNED_RANGE = enum.auto()
#
#
# class LossType(enum.Enum):
#     MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
#     RESCALED_MSE = (
#         enum.auto()
#     )  # use raw MSE loss (with RESCALED_KL when learning variances)
#     KL = enum.auto()  # use the variational lower-bound
#     RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
#
#     def is_vb(self):
#         return self == LossType.KL or self == LossType.RESCALED_KL
#
#
# class GaussianDiffusion:
#     """
#     Utilities for training and sampling diffusion models.
#
#     Ported directly from here, and then adapted over time to further experimentation.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
#
#     :param betas: a 1-D numpy array of betas for each diffusion timestep,
#                   starting at T and going to 1.
#     :param model_mean_type: a ModelMeanType determining what the model outputs.
#     :param model_var_type: a ModelVarType determining how variance is output.
#     :param loss_type: a LossType determining the loss function to use.
#     :param rescale_timesteps: if True, pass floating point timesteps into the
#                               model so that they are always scaled like in the
#                               original paper (0 to 1000).
#     """
#
#     def __init__(
#         self,
#         *,
#         betas,
#         model_mean_type,
#         model_var_type,
#         loss_type,
#         rescale_timesteps=False,
#
#     ):
#         self.model_mean_type = model_mean_type
#         self.model_var_type = model_var_type
#         self.loss_type = loss_type
#         self.rescale_timesteps = rescale_timesteps
#
#         # Use float64 for accuracy.
#         betas = np.array(betas, dtype=np.float64)
#         self.betas = betas
#         assert len(betas.shape) == 1, "betas must be 1-D"
#         assert (betas > 0).all() and (betas <= 1).all()
#
#         self.num_timesteps = int(betas.shape[0])
#
#         alphas = 1.0 - betas
#         self.alphas_cumprod = np.cumprod(alphas, axis=0)
#         self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
#         self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
#         assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
#
#         # for harmonizing
#         self.sqrt_alphas = np.sqrt(alphas)
#         self.sqrt_betas = np.sqrt(betas)
#
#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
#         self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
#         self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
#         self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
#
#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         self.posterior_variance = (
#             betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#         )
#         # log calculation clipped because the posterior variance is 0 at the
#         # beginning of the diffusion chain.
#         self.posterior_log_variance_clipped = np.log(
#             np.append(self.posterior_variance[1], self.posterior_variance[1:])
#         )
#         self.posterior_mean_coef1 = (
#             betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
#         )
#         self.posterior_mean_coef2 = (
#             (1.0 - self.alphas_cumprod_prev)
#             * np.sqrt(alphas)
#             / (1.0 - self.alphas_cumprod)
#         )
#
#     def q_mean_variance(self, x_start, t):
#         """
#         Get the distribution q(x_t | x_0).
#
#         :param x_start: the [N x C x ...] tensor of noiseless inputs.
#         :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
#         :return: A tuple (mean, variance, log_variance), all of x_start's shape.
#         """
#         mean = (
#             _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#         )
#         variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
#         log_variance = _extract_into_tensor(
#             self.log_one_minus_alphas_cumprod, t, x_start.shape
#         )
#         return mean, variance, log_variance
#
#     def q_sample(self, x_start, t, noise=None):
#         """
#         Diffuse the data for a given number of diffusion steps.
#
#         In other words, sample from q(x_t | x_0).
#
#         :param x_start: the initial data batch.
#         :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
#         :param noise: if specified, the split-out normal noise.
#         :return: A noisy version of x_start.
#         """
#         if noise is None:
#             noise = th.randn_like(x_start)
#         assert noise.shape == x_start.shape
#
#         return (
#             _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#             + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
#             * noise
#         )
#
#     def q_posterior_mean_variance(self, x_start, x_t, t):
#         """
#         Compute the mean and variance of the diffusion posterior:
#
#             q(x_{t-1} | x_t, x_0)
#
#         """
#         assert x_start.shape == x_t.shape
#         posterior_mean = (
#             _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
#             + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = _extract_into_tensor(
#             self.posterior_log_variance_clipped, t, x_t.shape
#         )
#         assert (
#             posterior_mean.shape[0]
#             == posterior_variance.shape[0]
#             == posterior_log_variance_clipped.shape[0]
#             == x_start.shape[0]
#         )
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped
#
#     def p_mean_variance(
#         self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,condition_args=None,original_args=None
#     ):
#         """
#         Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
#         the initial x, x_0.
#
#         :param model: the model, which takes a signal and a batch of timesteps
#                       as input.
#         :param x: the [N x C x ...] tensor at time t.
#         :param t: a 1-D Tensor of timesteps.
#         :param clip_denoised: if True, clip the denoised signal into [-1, 1].
#         :param denoised_fn: if not None, a function which applies to the
#             x_start prediction before it is used to sample. Applies before
#             clip_denoised.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#         :return: a dict with the following keys:
#                  - 'mean': the model mean output.
#                  - 'variance': the model variance output.
#                  - 'log_variance': the log of 'variance'.
#                  - 'pred_xstart': the prediction for x_0.
#         """
#         if model_kwargs is None:
#             model_kwargs = {}
#
#         B, C = x.shape[:2]
#         assert t.shape == (B,)
#
#         import pdb
#         # pdb.set_trace()
#         # breakpoint()
#
#         # model_output = model(x, self._scale_timesteps(t), **model_kwargs) # model output
#         # breakpoint()
#         # pdb.set_trace()
#
#         # condition_args.shape
#         # torch.Size([5, 1, 5, 32, 32])
#         #
#         # x.shape
#         # torch.Size([5, 1, 5, 32, 32])
#
#         model_output = model(x, self._scale_timesteps(t), condition_args) # model output
#         # breakpoint()
#
#         if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
#             assert model_output.shape == (B, C * 2, *x.shape[2:])
#             model_output, model_var_values = th.split(model_output, C, dim=1)
#             if self.model_var_type == ModelVarType.LEARNED:
#                 model_log_variance = model_var_values
#                 model_variance = th.exp(model_log_variance)
#             else:
#                 min_log = _extract_into_tensor(
#                     self.posterior_log_variance_clipped, t, x.shape
#                 )
#                 max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
#                 # The model_var_values is [-1, 1] for [min_var, max_var].
#                 frac = (model_var_values + 1) / 2
#                 model_log_variance = frac * max_log + (1 - frac) * min_log
#                 model_variance = th.exp(model_log_variance)
#         else:
#             model_variance, model_log_variance = {
#                 # for fixedlarge, we set the initial (log-)variance like so
#                 # to get a better decoder log likelihood.
#                 ModelVarType.FIXED_LARGE: (
#                     np.append(self.posterior_variance[1], self.betas[1:]),
#                     np.log(np.append(self.posterior_variance[1], self.betas[1:])),
#                 ),
#                 ModelVarType.FIXED_SMALL: (
#                     self.posterior_variance,
#                     self.posterior_log_variance_clipped,
#                 ),
#             }[self.model_var_type]
#             model_variance = _extract_into_tensor(model_variance, t, x.shape)
#             model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
#
#         def process_xstart(x):
#             if denoised_fn is not None:
#                 x = denoised_fn(x)
#             if clip_denoised:
#                 return x.clamp(-1, 1)
#             return x
#
#         if self.model_mean_type == ModelMeanType.PREVIOUS_X:
#             pred_xstart = process_xstart(
#                 self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
#             )
#             model_mean = model_output
#
#         elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
#             if self.model_mean_type == ModelMeanType.START_X:
#                 pred_xstart = process_xstart(model_output)
#             else:
#                 pred_xstart = process_xstart(
#                     self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
#                 )
#
#             model_mean, _, _ = self.q_posterior_mean_variance(
#                 x_start=pred_xstart, x_t=x, t=t
#             )
#         else:
#             raise NotImplementedError(self.model_mean_type)
#
#         assert (
#             model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
#         )
#         # breakpoint()
#         return {
#             "mean": model_mean,
#             "variance": model_variance,
#             "log_variance": model_log_variance,
#             "pred_xstart": pred_xstart,
#         }
#
#     def _predict_xstart_from_eps(self, x_t, t, eps):
#         assert x_t.shape == eps.shape
#         return (
#             _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
#             - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
#         )
#
#     def _predict_xstart_from_xprev(self, x_t, t, xprev):
#         assert x_t.shape == xprev.shape
#         return (  # (xprev - coef2*x_t) / coef1
#             _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
#             - _extract_into_tensor(
#                 self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
#             )
#             * x_t
#         )
#
#     def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
#         return (
#             _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
#             - pred_xstart
#         ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
#
#     def _scale_timesteps(self, t):
#         if self.rescale_timesteps:
#             return t.float() * (1000.0 / self.num_timesteps)
#         return t
#
#     def p_sample(
#         self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,condition_args=None,original_args=None
#     ):
#         """
#         Sample x_{t-1} from the model at the given timestep.
#
#         :param model: the model to sample from.
#         :param x: the current tensor at x_{t-1}.
#         :param t: the value of t, starting at 0 for the first diffusion step.
#         :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
#         :param denoised_fn: if not None, a function which applies to the
#             x_start prediction before it is used to sample.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#         :return: a dict containing the following keys:
#                  - 'sample': a random sample from the model.
#                  - 'pred_xstart': a prediction of x_0.
#         """
#         out = self.p_mean_variance(
#             model,
#             x,
#             t,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             model_kwargs=model_kwargs,
#             condition_args=condition_args,
#             original_args=original_args
#
#         )
#         noise = th.randn_like(x)
#
#         # breakpoint()
#         nonzero_mask = (
#             (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
#         )  # no noise when t == 0
#         # breakpoint()
#         # sample = out["mean"] +nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
#         sample = out["mean"] + th.exp(0.5 * out["log_variance"]) * noise
#         return {"sample": sample, "pred_xstart": out["pred_xstart"]}
#
#     def p_sample_loop(
#         self,
#         model,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#         cond_kwargs=None,
#         condition_args=None,
#         original_args=None
#     ):
#         """
#         Generate samples from the model.
#
#         :param model: the model module.
#         :param shape: the shape of the samples, (N, C, H, W).
#         :param noise: if specified, the noise from the encoder to sample.
#                       Should be of the same shape as `shape`.
#         :param clip_denoised: if True, clip x_start predictions to [-1, 1].
#         :param denoised_fn: if not None, a function which applies to the
#             x_start prediction before it is used to sample.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#         :param device: if specified, the device to create the samples on.
#                        If not specified, use a model parameter's device.
#         :param progress: if True, show a tqdm progress bar.
#         :return: a non-differentiable batch of samples.
#         """
#         final = None
#         # cond=condition_args
#         # breakpoint()
#         for sample in self.p_sample_loop_progressive(
#             model,
#             shape,
#             noise=noise,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             model_kwargs=model_kwargs,
#             device=device,
#             progress=progress,
#             cond_kwargs=cond_kwargs,
#             condition_args=condition_args,
#             original_args=original_args
#         ):
#             final = sample
#         return final["sample"]
#
#     def p_sample_loop_progressive(
#         self,
#         model,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#         cond_kwargs=None,
#         condition_args=None,
#         original_args=None
#     ):
#         """
#         Generate samples from the model and yield intermediate samples from
#         each timestep of diffusion.
#
#         Arguments are the same as p_sample_loop().
#         Returns a generator over dicts, where each dict is the return value of
#         p_sample().
#         """
#         if device is None:
#             device = next(model.parameters()).device
#         assert isinstance(shape, (tuple, list))
#
#         # breakpoint()
#         if noise is not None:
#             img = noise
#         else:
#             img = th.randn(*shape, device=device)
#         indices = list(range(self.num_timesteps))[::-1]
#         # breakpoint()
#         if progress:
#             # Lazy import so that we don't depend on tqdm.
#             from tqdm.auto import tqdm
#             indices = tqdm(indices)
#
#         resampling_steps = 1
#         cond_frames = cond_kwargs["cond_frames"]
#         if cond_frames:
#             cond_img = cond_kwargs["cond_img"]
#             resampling_steps = cond_kwargs["resampling_steps"]
#
#          #added code for plotting purpose
#         import os
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from matplotlib import gridspec
#
#         import shutil
#
#         # Delete the "figures" directory and its contents (if it exists)
#         if os.path.exists('figures_samples'):
#              shutil.rmtree('figures_samples')
#         #
#         # # Create a new "figures" directory
#         os.makedirs('figures_samples')
#         # end of the added code
#         for i in indices:
#             # breakpoint()
#
#             # pdb.set_trace()
#             t = th.tensor([i] * shape[0], device=device)
#             # pdb.set_trace()
#             # breakpoint()
#
#             for r in range(resampling_steps):
#                 if cond_frames:
#                     img[:,:,cond_frames] = cond_img
#
#                 with th.no_grad():
#                     out = self.p_sample(
#                         model,
#                         img,
#                         t,
#                         clip_denoised=clip_denoised,
#                         denoised_fn=denoised_fn,
#                         model_kwargs=model_kwargs,
#                         condition_args=condition_args,
#                         original_args=original_args
#                     )
#
#                 img = out["sample"]
#                 # img_samp=img.permute(0, 2, 3, 4, 1)
#                 # condition_samp=condition_args.permute(0,2,3,4,1)
#                 # original_samp=original_args.permute(0,2,3,4,1)
#                 # # breakpoint()
#                 # if i%50==0:
#                 # # Create a figure with 10 subplots (one for each frame)
#                 #     fig = plt.figure(figsize=(20, 10))
#                 #     gs = gridspec.GridSpec(nrows=3, ncols=img_samp.shape[1], left=0, right=1, top=1, bottom=0)
#                 #     plt.subplots_adjust(wspace=0, hspace=0)
#                 #     # Loop over the frames
#                 #     for j in range(img_samp[0].shape[0]):
#                 #         # Select the current frame
#                 #         frame_generated = img_samp[0][j, :, :, :].cpu()
#                 #         frame_shuffled = condition_samp[0][j, :, :, :].cpu()
#                 #         frame_ground_truth = original_samp[0][j, :, :, :].cpu()
#                 #
#                 #         # Create a subplot at the current position
#                 #         ax1 = fig.add_subplot(gs[0, j])
#                 #         ax2 = fig.add_subplot(gs[1, j])
#                 #         ax3 = fig.add_subplot(gs[2, j])
#                 #
#                 #         # Plot the frame in grayscale
#                 #
#                 #         ax1.imshow(frame_shuffled, cmap='gray')
#                 #         ax1.axis('off')
#                 #         if j==9:
#                 #             ax1.set_title('shuffled samples', loc='center', fontsize=25)
#                 #
#                 #         ax2.imshow(frame_ground_truth, cmap='gray')
#                 #         ax2.axis('off')
#                 #         if j==9:
#                 #             ax2.set_title('ground truth samples', loc='center', fontsize=25)
#                 #         ax3.imshow(frame_generated, cmap='gray')
#                 #         ax3.axis('off')
#                 #         if j==9:
#                 #             ax3.set_title('generated samples', loc='center', fontsize=25)
#                 #
#                 #     plt.savefig('figures_samples/batch_{}.png'.format(i))
#                 #     breakpoint()
#                 if r < resampling_steps - 1:
#                     img = self.forward_diffusion(img, device, i)
#
#             yield out
#
#     def forward_diffusion(
#         self,
#         x_start,
#         device,
#         i,
#         ):
#         """ starting from x_i, create a sample after length forward diffusion steps
#             i.e. x_t+1 ~ q(x_t+1|x_t)
#
#         Args:
#             x_start: the tensor at time x_i
#             device: gpu or cpu device on which inference is done
#             i: starting time of the diffusion process
#             length: amount of diffusion steps
#
#         Returns:
#             tensor sample x_t+1
#         """
#
#         noise = th.randn_like(x_start, device=device)
#         t = th.tensor([i] * x_start.shape[0], device=device)
#         x_t = (_extract_into_tensor(self.sqrt_alphas, t, x_start.shape) * x_start
#         + _extract_into_tensor(self.sqrt_betas, t, x_start.shape)
#         * noise)
#
#         return x_t
#
#     def resampling(
#         self,
#         x_start,
#         model,
#         clip_denoised,
#         denoised_fn,
#         model_kwargs,
#         device,
#         t,
#         jump_length,
#         use_ddim,
#         ):
#
#         sample_fn = self.p_sample if not use_ddim else self.ddim_sample
#         indices = list(range(t, t + jump_length))
#
#         x_t = x_start
#         for i in indices:
#             x_t = self.forward_diffusion(x_t, device, i)
#
#         for i in indices[::-1]:
#             t = th.tensor([i] * x_t.shape[0], device=device)
#             with th.no_grad():
#                 out = sample_fn(
#                             model,
#                             x_t,
#                             t,
#                             clip_denoised=clip_denoised,
#                             denoised_fn=denoised_fn,
#                             model_kwargs=model_kwargs,
#                         )
#             x_t = out['sample']
#         return x_t
#
#     def ddim_sample(
#         self,
#         model,
#         x,
#         t,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         eta=0.0,
#         condition_args=None,
#         original_args=None
#     ):
#         """
#         Sample x_{t-1} from the model using DDIM.
#
#         Same usage as p_sample().
#         """
#         out = self.p_mean_variance(
#             model,
#             x,
#             t,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             model_kwargs=model_kwargs,
#             condition_args=condition_args,
#             original_args=original_args
#         )
#         # Usually our model outputs epsilon, but we re-derive it
#         # in case we used x_start or x_prev prediction.
#
#         eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
#         alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
#         alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
#         sigma = (
#             eta
#             * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
#             * th.sqrt(1 - alpha_bar / alpha_bar_prev)
#         )
#         # Equation 12.
#         noise = th.randn_like(x)
#         mean_pred = (
#             out["pred_xstart"] * th.sqrt(alpha_bar_prev)
#             + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
#         )
#         nonzero_mask = (
#             (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
#         )  # no noise when t == 0
#         sample = mean_pred + nonzero_mask * sigma * noise
#
#         return {"sample": sample, "pred_xstart": out["pred_xstart"]}
#
#     def ddim_reverse_sample(
#         self,
#         model,
#         x,
#         t,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         eta=0.0,
#     ):
#         """
#         Sample x_{t+1} from the model using DDIM reverse ODE.
#         """
#         assert eta == 0.0, "Reverse ODE only for deterministic path"
#         out = self.p_mean_variance(
#             model,
#             x,
#             t,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             model_kwargs=model_kwargs,
#         )
#         # Usually our model outputs epsilon, but we re-derive it
#         # in case we used x_start or x_prev prediction.
#         eps = (
#             _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
#             - out["pred_xstart"]
#         ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
#         alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
#
#         # Equation 12. reversed
#         mean_pred = (
#             out["pred_xstart"] * th.sqrt(alpha_bar_next)
#             + th.sqrt(1 - alpha_bar_next) * eps
#         )
#
#         return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
#
#     def ddim_sample_loop(
#         self,
#         model,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#         eta=0.0,
#         cond_kwargs=None
#     ):
#         """
#         Generate samples from the model using DDIM.
#
#         Same usage as p_sample_loop().
#         """
#         final = None
#         for sample in self.ddim_sample_loop_progressive(
#             model,
#             shape,
#             noise=noise,
#             clip_denoised=clip_denoised,
#             denoised_fn=denoised_fn,
#             model_kwargs=model_kwargs,
#             device=device,
#             progress=progress,
#             eta=eta,
#             cond_kwargs=cond_kwargs
#         ):
#             final = sample
#         return final["sample"]
#
#     def ddim_sample_loop_progressive(
#         self,
#         model,
#         shape,
#         noise=None,
#         clip_denoised=True,
#         denoised_fn=None,
#         model_kwargs=None,
#         device=None,
#         progress=False,
#         eta=0.0,
#         cond_kwargs=None
#     ):
#         """
#         Use DDIM to sample from the model and yield intermediate samples from
#         each timestep of DDIM.
#
#         Same usage as p_sample_loop_progressive().
#         """
#         if device is None:
#             device = next(model.parameters()).device
#         assert isinstance(shape, (tuple, list))
#         if noise is not None:
#             img = noise
#         else:
#             img = th.randn(*shape, device=device)
#         indices = list(range(self.num_timesteps))[::-1]
#
#         if progress:
#             # Lazy import so that we don't depend on tqdm.
#             from tqdm.auto import tqdm
#
#             indices = tqdm(indices)
#
#         saver = cond_kwargs["saver"]
#         resampling_steps = 1
#         cond_frames = cond_kwargs["cond_frames"]
#         if cond_frames:
#             cond_img = cond_kwargs["cond_img"]
#             resampling_steps = cond_kwargs["resampling_steps"]
#
#         for i in indices:
#             t = th.tensor([i] * shape[0], device=device)
#
#             for r in range(resampling_steps):
#                 if cond_frames:
#                     img[:,:,cond_frames] = cond_img
#
#                     with th.no_grad():
#                         out = self.ddim_sample(
#                             model,
#                             img,
#                             t,
#                             clip_denoised=clip_denoised,
#                             denoised_fn=denoised_fn,
#                             model_kwargs=model_kwargs,
#                             eta=eta,
#                         )
#
#                     img = out["sample"]
#
#                     if r < resampling_steps - 1:
#                         img = self.forward_diffusion(img, device, i)
#
#                 yield out
#
#
#     def _vb_terms_bpd(
#         self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, masks=[]
#     ):
#         """
#         Get a term for the variational lower-bound.
#
#         The resulting units are bits (rather than nats, as one might expect).
#         This allows for comparison to other papers.
#
#         :return: a dict with the following keys:
#                  - 'output': a shape [N] tensor of NLLs or KLs.
#                  - 'pred_xstart': the x_0 predictions.
#         """
#         true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
#             x_start=x_start, x_t=x_t, t=t
#         )
#         out = self.p_mean_variance(
#             model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
#         )
#         kl = normal_kl(
#             true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
#         )
#         for b, mask in enumerate(masks):
#             kl[b, :, mask] = th.zeros_like(kl[b, :, mask])
#         kl = mean_flat(kl) / np.log(2.0)
#
#         decoder_nll = -discretized_gaussian_log_likelihood(
#             x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
#         )
#         assert decoder_nll.shape == x_start.shape
#         decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
#
#         # At the first timestep return the decoder NLL,
#         # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
#         output = th.where((t == 0), decoder_nll, kl)
#         return {"output": output, "pred_xstart": out["pred_xstart"]}
#
#
#     def _vb_terms_bpd_unshuffle(
#         self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
#     ):
#         """
#         Get a term for the variational lower-bound.
#
#         The resulting units are bits (rather than nats, as one might expect).
#         This allows for comparison to other papers.
#
#         :return: a dict with the following keys:
#                  - 'output': a shape [N] tensor of NLLs or KLs.
#                  - 'pred_xstart': the x_0 predictions.
#         """
#         true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
#             x_start=x_start, x_t=x_t, t=t
#         )
#         out = self.p_mean_variance(
#             model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
#         )
#         kl = normal_kl(
#             true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
#         )
#         # for b, mask in enumerate(masks):
#         #     kl[b, :, mask] = th.zeros_like(kl[b, :, mask])
#         # kl = mean_flat(kl) / np.log(2.0)
#
#         decoder_nll = -discretized_gaussian_log_likelihood(
#             x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
#         )
#         assert decoder_nll.shape == x_start.shape
#         decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
#
#         # At the first timestep return the decoder NLL,
#         # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
#         output = th.where((t == 0), decoder_nll, kl)
#         return {"output": output, "pred_xstart": out["pred_xstart"]}
#
#     def training_losses(
#         self,
#         model,
#         x_start,
#         t,
#         model_kwargs=None,
#         noise=None,
#         max_num_mask_frames=4,
#         mask_range=None,
#         uncondition_rate=0,
#         exclude_conditional=True
#         ):
#         """
#         Compute training losses for a single timestep.
#
#         :param model: the model to evaluate loss on.
#         :param x_start: the [N x C x ...] tensor of inputs.
#         :param t: a batch of timestep indices.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#         :param noise: if specified, the specific Gaussian noise to try to remove.
#         :return: a dict with the key "loss" containing a tensor of shape [N].
#                  Some mean or variance settings may also have other keys.
#         """
#         if model_kwargs is None:
#             model_kwargs = {}
#         if noise is None:
#             noise = th.randn_like(x_start)
#         x_t = self.q_sample(x_start, t, noise=noise)
#
#         masks = th.ones_like(x_t)
#         import pdb
#         # pdb.set_trace()
#
#         for b in range(x_t.shape[0]):
#             unconditional = np.random.binomial(1, uncondition_rate)
#             if unconditional:
#                 mask = []
#             else:
#                 r = random.randint(1, max_num_mask_frames)
#                 mask = random.sample(range(*mask_range), r)
#
#             masks[b, :, mask] = 0
#
#         noise *= masks
#         x_t = x_t * masks + (1 - masks) * x_start
#         terms = {}
#
#         if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
#             model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
#             # model_output = model(x_t, self._scale_timesteps(t), condition_data)
#
#             if self.model_var_type in [
#                 ModelVarType.LEARNED,
#                 ModelVarType.LEARNED_RANGE,
#             ]:
#                 B, C = x_t.shape[:2]
#                 assert model_output.shape == (B, C * 2, *x_t.shape[2:])
#                 model_output, model_var_values = th.split(model_output, C, dim=1)
#                 # Learn the variance using the variational bound, but don't let
#                 # it affect our mean prediction.
#                 frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
#                 terms["vb"] = self._vb_terms_bpd(
#                     model=lambda *args, r=frozen_out: r,
#                     x_start=x_start,
#                     x_t=x_t,
#                     t=t,
#                     clip_denoised=True,
#                     masks=masks
#                 )["output"]
#                 if self.loss_type == LossType.RESCALED_MSE:
#                     # Divide by 1000 for equivalence with initial implementation.
#                     # Without a factor of 1/1000, the VB term hurts the MSE term.
#                     terms["vb"] *= self.num_timesteps / 1000.0
#
#             target = {
#                 ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
#                     x_start=x_start, x_t=x_t, t=t
#                 )[0],
#                 ModelMeanType.START_X: x_start,
#                 ModelMeanType.EPSILON: noise,
#             }[self.model_mean_type]
#             assert model_output.shape == target.shape == x_start.shape
#
#             mse = (target - model_output) ** 2
#             if exclude_conditional:
#                 mse *= masks
#                 terms["mse"] = sum_flat(mse) / sum_flat(masks)
#             else:
#                 terms["mse"] = mean_flat(mse)
#
#             if "vb" in terms:
#                 terms["loss"] = terms["mse"] + terms["vb"]
#             else:
#                 terms["loss"] = terms["mse"]
#         else:
#             raise NotImplementedError(self.loss_type)
#
#         return terms
#
#     def training_losses2(
#             self,
#             model,
#             x_start,
#             t,
#             condition_data=None,
#             model_kwargs=None,
#             noise=None,
#             max_num_mask_frames=4,
#             mask_range=None,
#             uncondition_rate=0,
#             exclude_conditional=True
#     ):
#         """
#         Compute training losses for a single timestep.
#
#         :param model: the model to evaluate loss on.
#         :param x_start: the [N x C x ...] tensor of inputs.
#         :param t: a batch of timestep indices.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#         :param noise: if specified, the specific Gaussian noise to try to remove.
#         :return: a dict with the key "loss" containing a tensor of shape [N].
#                  Some mean or variance settings may also have other keys.
#         """
#         if noise is None:
#             noise = th.randn_like(x_start)
#         x_t = self.q_sample(x_start, t, noise=noise)
#         # for b in range(x_t.shape[0]):
#         #     unconditional = np.random.binomial(1, uncondition_rate)
#         #     if unconditional:
#         #         mask = []
#         #     else:
#         #         r = random.randint(1, max_num_mask_frames)
#         #         mask = random.sample(range(*mask_range), r)
#         #
#         #     masks[b, :, mask] = 0
#
#         # noise *= masks
#         # x_t = x_t * masks + (1 - masks) * x_start
#         terms = {}
#         import pdb
#         # pdb.set_trace()
#         if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
#             # breakpoint()
#             model_output = model(x_t, self._scale_timesteps(t), condition_data)
#
#             if self.model_var_type in [
#                 ModelVarType.LEARNED,
#                 ModelVarType.LEARNED_RANGE,
#             ]:
#                 B, C = x_t.shape[:2]
#                 assert model_output.shape == (B, C * 2, *x_t.shape[2:])
#                 model_output, model_var_values = th.split(model_output, C, dim=1)
#                 # Learn the variance using the variational bound, but don't let
#                 # it affect our mean prediction.
#                 frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
#                 terms["vb"] = self._vb_terms_bpd_unshuffle(
#                     model=lambda *args, r=frozen_out: r,
#                     x_start=x_start,
#                     x_t=x_t,
#                     t=t,
#                     clip_denoised=True
#                 )["output"]
#                 if self.loss_type == LossType.RESCALED_MSE:
#                     # Divide by 1000 for equivalence with initial implementation.
#                     # Without a factor of 1/1000, the VB term hurts the MSE term.
#                     terms["vb"] *= self.num_timesteps / 1000.0
#
#             target = {
#                 ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
#                     x_start=x_start, x_t=x_t, t=t
#                 )[0],
#                 ModelMeanType.START_X: x_start,
#                 ModelMeanType.EPSILON: noise,
#             }[self.model_mean_type]
#             assert model_output.shape == target.shape == x_start.shape
#
#             mse = (target - model_output) ** 2
#             if exclude_conditional:
#                 # mse *= masks
#                 terms["mse"] = sum_flat(mse) #/ sum_flat(masks)
#             else:
#                 terms["mse"] = mean_flat(mse)
#
#             if "vb" in terms:
#                 terms["loss"] = terms["mse"] + terms["vb"]
#             else:
#                 terms["loss"] = terms["mse"]
#         else:
#             raise NotImplementedError(self.loss_type)
#
#         return terms
#
#     def _prior_bpd(self, x_start):
#         """
#         Get the prior KL term for the variational lower-bound, measured in
#         bits-per-dim.
#
#         This term can't be optimized, as it only depends on the encoder.
#
#         :param x_start: the [N x C x ...] tensor of inputs.
#         :return: a batch of [N] KL values (in bits), one per batch element.
#         """
#         batch_size = x_start.shape[0]
#         t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
#         qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
#         kl_prior = normal_kl(
#             mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
#         )
#         return mean_flat(kl_prior) / np.log(2.0)
#
#     def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None, progress=True):
#         """
#         Compute the entire variational lower-bound, measured in bits-per-dim,
#         as well as other related quantities.
#
#         :param model: the model to evaluate loss on.
#         :param x_start: the [N x C x ...] tensor of inputs.
#         :param clip_denoised: if True, clip denoised samples.
#         :param model_kwargs: if not None, a dict of extra keyword arguments to
#             pass to the model. This can be used for conditioning.
#
#         :return: a dict containing the following keys:
#                  - total_bpd: the total variational lower-bound, per batch element.
#                  - prior_bpd: the prior term in the lower-bound.
#                  - vb: an [N x T] tensor of terms in the lower-bound.
#                  - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
#                  - mse: an [N x T] tensor of epsilon MSEs for each timestep.
#         """
#         device = x_start.device
#         batch_size = x_start.shape[0]
#
#         vb = []
#         xstart_mse = []
#         mse = []
#         indices = list(range(self.num_timesteps))[::-1]
#         if progress:
#             from tqdm.auto import tqdm
#             indices = tqdm(indices)
#
#         for t in indices:
#             t_batch = th.tensor([t] * batch_size, device=device)
#             noise = th.randn_like(x_start)
#             x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
#             # Calculate VLB term at the current timestep
#             with th.no_grad():
#                 out = self._vb_terms_bpd(
#                     model,
#                     x_start=x_start,
#                     x_t=x_t,
#                     t=t_batch,
#                     clip_denoised=clip_denoised,
#                     model_kwargs=model_kwargs,
#                 )
#             vb.append(out["output"])
#             xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
#             eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
#             mse.append(mean_flat((eps - noise) ** 2))
#
#         vb = th.stack(vb, dim=1)
#         xstart_mse = th.stack(xstart_mse, dim=1)
#         mse = th.stack(mse, dim=1)
#
#         prior_bpd = self._prior_bpd(x_start)
#         total_bpd = vb.sum(dim=1) + prior_bpd
#         return {
#             "total_bpd": total_bpd,
#             "prior_bpd": prior_bpd,
#             "vb": vb,
#             "xstart_mse": xstart_mse,
#             "mse": mse,
#         }
#
#
# def _extract_into_tensor(arr, timesteps, broadcast_shape):
#     """
#     Extract values from a 1-D numpy array for a batch of indices.
#
#     :param arr: the 1-D numpy array.
#     :param timesteps: a tensor of indices into the array to extract.
#     :param broadcast_shape: a larger shape of K dimensions with the batch
#                             dimension equal to the length of timesteps.
#     :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
#     """
#     res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
#     while len(res.shape) < len(broadcast_shape):
#         res = res[..., None]
#     return res.expand(broadcast_shape)
