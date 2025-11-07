# This scheduler file is a derivative work.
# It originates from Diffusers (Copyright Katherine Crowson and The HuggingFace Team, Apache 2.0 License)
# and has been modified to include a "Restoration Guide" logic.
#
# The restoration logic is ported from the original SUPIR project's EDM sampler,
# which is Copyright (c) 2024 by SupPixel Pty Ltd.
#
# Modifications and adaptations to this file are Copyright (c) 2025 by DEVAIEXP.
#
# Because this file incorporates a key functional component from the SUPIR project,
# its use is governed by the SUPIR Software License Agreement.
# Use of this scheduler is therefore strictly limited to NON-COMMERCIAL purposes.
#
# For the full license terms, please see the LICENSE_SUPIR.md file in the
# parent pipeline directory.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.utils.torch_utils import randn_tensor


if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerDiscrete
class EulerDiscreteRestoreSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output, including guided prediction.

    Args:
        prev_sample (`torch.Tensor`): Computed sample `(x_{t-1})` of previous timestep.
        pred_original_sample (`torch.Tensor`, *optional*): Predicted denoised sample `(x_{0})`.
        guided_pred_original_sample (`torch.Tensor`, *optional*): Predicted denoised sample `(x_{0})` after guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    guided_pred_original_sample: Optional[torch.Tensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class EulerDiscreteRestoreScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler, adapted for SUPIR restoration guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray` or `List[float]`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        interpolation_type (`str`, defaults to `"linear"`, *optional*):
            The interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be one of
            `"linear"` or `"log_linear"`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        use_beta_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use beta sigmas for step sizes in the noise schedule during the sampling process. Refer to [Beta
            Sampling is All You Need](https://huggingface.co/papers/2407.12173) for more information.
        sigma_min (`float`, *optional*):
            Minimum sigma value for Karras sigmas or exponential sigmas. If `None`, it will be inferred from `in_sigmas`
            depending on `self.config.interpolation_type`.
        sigma_max (`float`, *optional*):
            Maximum sigma value for Karras sigmas or exponential sigmas. If `None`, it will be inferred from `in_sigmas`
            depending on `self.config.interpolation_type`.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        timestep_type (`str`, defaults to `"discrete"`):
            The type of timesteps used by the scheduler. Can be `"discrete"` or `"continuous"`.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
        restore_cfg_s_tmin (`float`, *optional*, defaults to 0.05):
            Sigma threshold below which restoration guidance is not applied. This is a custom parameter for restoration.
        sigma_max_for_guidance (`float`, *optional*, defaults to 14.6146):
            Reference maximum sigma used in the restoration guidance weight calculation. This is a custom parameter for restoration.
        scale_linear_exponent (`float`, *optional*, defaults to 2.0):
            Exponent for the scale linear beta schedule. This is a custom parameter for restoration.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
        restore_cfg_s_tmin: float = 0.05,
        scale_linear_exponent: float = 2.0,
        sigma_max_for_guidance: float = 14.6146,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if (
            sum(
                [
                    self.config.use_beta_sigmas,
                    self.config.use_exponential_sigmas,
                    self.config.use_karras_sigmas,
                ]
            )
            > 1
        ):
            raise ValueError("Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used.")
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** scale_linear_exponent
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # setable values
        self.num_inference_steps = None

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if timestep_type == "continuous" and prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas])
        else:
            self.timesteps = timesteps

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.is_scale_input_called = False
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        max_sigma = max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma

        return (max_sigma**2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to support arbitrary timesteps schedule schedule. If `None`, timesteps and sigmas
                will be generated based on the relevant scheduler attributes. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`, and the timesteps will be generated based on the
                custom sigmas schedule.
        """

        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` should be set.")
        if num_inference_steps is None and timesteps is None and sigmas is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps` or `sigmas.")
        if num_inference_steps is not None and (timesteps is not None or sigmas is not None):
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps` or `sigmas`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_karras_sigmas = True`.")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")
        if timesteps is not None and self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            raise ValueError("Cannot set `timesteps` with `config.timestep_type = 'continuous'` and `config.prediction_type = 'v_prediction'`.")

        if num_inference_steps is None:
            num_inference_steps = len(timesteps) if timesteps is not None else len(sigmas) - 1
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            log_sigmas = np.log(np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5))
            sigmas = np.array(sigmas).astype(np.float32)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas[:-1]])

        else:
            if timesteps is not None:
                timesteps = np.array(timesteps).astype(np.float32)
            else:
                # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
                if self.config.timestep_spacing == "linspace":
                    timesteps = np.linspace(
                        0,
                        self.config.num_train_timesteps - 1,
                        num_inference_steps,
                        dtype=np.float32,
                    )[::-1].copy()
                elif self.config.timestep_spacing == "leading":
                    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                    timesteps += self.config.steps_offset
                elif self.config.timestep_spacing == "trailing":
                    step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                    timesteps -= 1
                else:
                    raise ValueError(f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'.")

            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            if self.config.interpolation_type == "linear":
                sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            elif self.config.interpolation_type == "log_linear":
                sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()
            else:
                raise ValueError(f"{self.config.interpolation_type} is not implemented. Please specify interpolation_type to either 'linear' or 'log_linear'")

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            if self.config.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            elif self.config.final_sigmas_type == "zero":
                sigma_last = 0
            else:
                raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")

            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas[:-1]]).to(device=device)
        else:
            self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from https://github.com/crowsonkb/k-diffusion/blob/686dbad0f39640ea25c8a8c6a6e56bb40eacefa2/k_diffusion/sampling.py#L17
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from https://github.com/crowsonkb/k-diffusion/blob/686dbad0f39640ea25c8a8c6a6e56bb40eacefa2/k_diffusion/sampling.py#L26
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(
        self,
        in_sigmas: torch.Tensor,
        num_inference_steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
    ) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [scipy.stats.beta.ppf(timestep, alpha, beta) for timestep in 1 - np.linspace(0, 1, num_inference_steps)]
            ]
        )
        return sigmas

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        z_center: Optional[torch.Tensor] = None,
        restoration_scale: float = 0.0,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteRestoreSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise). Includes Karras noise schedule
        and Restoration Guidance.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model (usually predicted noise, epsilon).
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain. This is one of the values from `self.timesteps`.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process (x_t). This should be the sample
                *before* `scale_model_input` is applied.
            z_center (`torch.Tensor`, *optional*):
                The latent representation of the center/LQ image for restoration guidance.
                Defaults to `None`.
            restoration_scale (`float`, *optional*, defaults to 0.0):
                Strength of the restoration guidance. If 0.0 or less, guidance is disabled.
            s_churn (`float`, *optional*, defaults to 0.0):
                Parameter for Karras schedule: stochasticity strength. 0 means deterministic.
            s_tmin (`float`, *optional*, defaults to 0.0):
                Parameter for Karras schedule: sigma threshold below which `s_churn` is not applied.
            s_tmax (`float`, *optional*, defaults to `float("inf")`):
                Parameter for Karras schedule: sigma threshold above which `s_churn` is not applied.
            s_noise (`float`, *optional*, defaults to 1.0):
                Parameter for Karras schedule: noise scaling factor when `s_churn > 0`.
            generator (`torch.Generator`, *optional*):
                A random number generator for Karras noise.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`EulerDiscreteRestoreSchedulerOutput`] or `tuple`.

        Returns:
            [`EulerDiscreteRestoreSchedulerOutput`] or `tuple`:
                If `return_dict` is `True`, `EulerDiscreteRestoreSchedulerOutput` is returned, otherwise a `tuple`
                is returned where the first element is the sample tensor (`prev_sample`).
        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample_orig_dtype = sample.dtype
        # Upcast sample para float32 ANTES de adicionar ruído Karras
        sample_float = sample.to(torch.float32)

        sigmas_on_device = self.sigmas.to(sample.device)
        sigma = sigmas_on_device[self.step_index]  # Sigma atual
        sigma = sigma.to(torch.float32)  # Garantir float32

        gamma = min(s_churn / (len(sigmas_on_device) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=model_output.device,
                generator=generator,
            )
            eps = noise * s_noise
            sample_float = sample_float + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample_float - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample_float / (sigma**2 + 1))
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

        # 2. Apply Restoration Guide
        guided_pred_original_sample = pred_original_sample  # Default
        sigma_orig = sigmas_on_device[self._step_index].to(dtype=torch.float32)
        restore_threshold = getattr(self.config, "restore_cfg_s_tmin", 0.05)
        apply_restoration = restoration_scale > 0.0 and z_center is not None and sigma_orig > restore_threshold

        if apply_restoration:
            z_center_float = z_center.to(dtype=pred_original_sample.dtype, device=pred_original_sample.device)
            sigma_T_ref = getattr(self.config, "sigma_max_for_guidance", sigmas_on_device[0].item())
            sigma_T_tensor = torch.tensor(sigma_T_ref, device=sigma_orig.device, dtype=torch.float32)
            if sigma_T_tensor > 1e-9:
                sigma_orig_safe = torch.clamp(sigma_orig, min=1e-9)
                k_val = (sigma_orig_safe / sigma_T_tensor) ** restoration_scale
                k = torch.clamp(k_val, 0.0, 1.0)
            else:
                k = torch.tensor(0.0, device=pred_original_sample.device, dtype=torch.float32)

            guided_pred_original_sample = (1.0 - k) * pred_original_sample + k * z_center_float

        # 3. Convert to an ODE derivative
        derivative = (sample_float - guided_pred_original_sample) / sigma_hat

        dt = sigmas_on_device[self.step_index + 1].to(torch.float32) - sigma_hat

        prev_sample = sample_float + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype).to(sample_orig_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, pred_original_sample.to(sample_orig_dtype))

        return EulerDiscreteRestoreSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample.to(sample_orig_dtype),
            guided_pred_original_sample=guided_pred_original_sample.to(sample_orig_dtype),
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if isinstance(timesteps, int) or isinstance(timesteps, torch.IntTensor) or isinstance(timesteps, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if sample.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timesteps = timesteps.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timesteps = timesteps.to(sample.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        alphas_cumprod = self.alphas_cumprod.to(sample)
        sqrt_alpha_prod = alphas_cumprod[step_indices] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[step_indices]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
