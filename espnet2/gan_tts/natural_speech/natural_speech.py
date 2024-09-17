from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, Dict, Optional

import torch
from typeguard import typechecked

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.gan_tts.utils import get_segments
from espnet2.gan_tts.natural_speech.generator import NSGenerator
from espnet2.gan_tts.natural_speech.loss import KLDivergenceLoss
from espnet2.torch_utils.device_funcs import force_gatherable

AVAILABLE_GENERATORS = {
    "ns_generator": NSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
}

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class NaturalSpeech(AbsGANTTS):
    @typechecked
    def __init__(self,
                 idim: int,
                 odim: int,
                 generator_type: str,
                 generator_params: Dict[str, Any],
                 discriminator_type: str,
                 discriminator_params: Dict[str, Any],
                 generator_adv_loss_params: Dict[str, Any],
                 discriminator_adv_loss_params: Dict[str, Any],
                 feat_match_loss_params: Dict[str, Any],
                 mel_loss_params: Dict[str, Any],
                 lambda_adv: float,
                 lambda_mel: float,
                 lambda_feat_match: float,
                 lambda_dur: float,
                 lambda_kl: float,
                 ) -> None:
        super().__init__()
        # define modules
        generator_class = AVAILABLE_GENERATORS[generator_type]
        if generator_type == 'ns_generator':
            generator_params.update(vocabs=idim,
                                    aux_channels=odim)
        self.generator = generator_class(**generator_params)
        dicsriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = dicsriminator_class(**discriminator_params)
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params)
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params)
        self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params)
        self.mel_loss = MelSpectrogramLoss(**mel_loss_params)
        