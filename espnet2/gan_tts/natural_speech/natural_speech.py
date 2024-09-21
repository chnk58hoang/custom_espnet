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
from espnet2.gan_tts.natural_speech.loss import SoftDTWKLLoss
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
                 sampling_rate,
                 generator_type: str,
                 generator_params: Dict[str, Any],
                 discriminator_type: str,
                 discriminator_params: Dict[str, Any],
                 generator_adv_loss_params: Dict[str, Any],
                 discriminator_adv_loss_params: Dict[str, Any],
                 feat_match_loss_params: Dict[str, Any],
                 mel_loss_params: Dict[str, Any],
                 soft_dtw_kl_params: Dict[str, Any],
                 lambda_adv: float,
                 lambda_adv_e2e: float,
                 lambda_mel: float,
                 lambda_feat_match: float,
                 lambda_dur: float,
                 lambda_fw_kl: float,
                 lambda_bw_kl: float,
                 cache_generator_outputs: bool = True,
                 plot_pred_mos: bool = False,
                 mos_pred_tool: str = "utmos",
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
        self.soft_dtw_kl = SoftDTWKLLoss(**soft_dtw_kl_params)
        self.lambda_adv = lambda_adv
        self.lambda_adv_e2e = lambda_adv_e2e
        self.lambda_mel = lambda_mel
        self.lambda_fw_kl = lambda_fw_kl
        self.lambda_bw_kl = lambda_bw_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur
        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator.spks
        self.langs = self.generator.langs
        self.spk_embed_dim = self.generator.spk_embed_dim

        # plot pseudo mos during training
        self.plot_pred_mos = plot_pred_mos
        if plot_pred_mos:
            if mos_pred_tool == "utmos":
                # Load predictor for UTMOS22 (https://arxiv.org/abs/2204.02152)
                self.predictor = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0", "utmos22_strong"
                )
            else:
                raise NotImplementedError(
                    f"Not supported mos_pred_tool: {mos_pred_tool}"
                )

    @property
    def require_raw_speech(self):
        """Return whether or not speech is required."""
        return True

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False

    def _forward_generator(self,
                           text: torch.Tensor,
                           text_lengths: torch.Tensor,
                           feats: torch.Tensor,
                           feats_lengths: torch.Tensor,
                           speech: torch.Tensor,
                           speech_lengths: torch.Tensor,
                           sids: Optional[torch.Tensor] = None,
                           spembs: Optional[torch.Tensor] = None,
                           lids: Optional[torch.Tensor] = None
                           ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)  # batch, channel, t_feat
        speech = speech.unsqueeze(1)  # batch, 1, t_wav
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache
        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        (pre_wav_post, pre_wav_e2e, z_start_idxs, z_q_start_idxs,
         x_mask, y_mask, frame_mask, pred_logdur,
         gt_logdur, (z, z_p, z_q, m_p, logs_p, m_q, logs_q)) = outs
        wav_post_segs = get_segments(x=speech,
                                     start_idxs=z_start_idxs * self.generator.upsample_factor,
                                     segment_size=self.generator.segment_size * self.generator.upsample_factor)
        wav_e2e_segs = get_segments(x=speech,
                                    start_idxs=z_q_start_idxs * self.generator.upsample_factor,
                                    segment_size=self.generator.segment_size * self.generator.upsample_factor)

        p_pre_wav_post = self.discriminator(pre_wav_post)
        p_pre_wav_e2e = self.discriminator(pre_wav_e2e)

        with torch.no_grad():
            p_wav_post = self.discriminator(wav_post_segs)
            p_wav_e2e = self.discriminator(wav_e2e_segs)

        with autocast(enabled=False):
            mel_loss = self.mel_loss(pre_wav_post, wav_post_segs)
            bw_kl_loss = self.soft_dtw_kl.forward(z_p, logs_q, m_p, logs_p, frame_mask, y_mask)
            fw_kl_loss = self.soft_dtw_kl.forward(z_q, logs_p, m_q, logs_q, y_mask, frame_mask)
            adv_loss = self.generator_adv_loss(p_pre_wav_post)
            adv_loss_e2e = self.generator_adv_loss(p_pre_wav_e2e)
            fm_loss = self.feat_match_loss(p_pre_wav_post, p_wav_post)
            dur_loss = torch.sum((gt_logdur - pred_logdur) ** 2, [1, 2]) / torch.sum(x_mask)
            dur_loss = torch.sum(dur_loss.float())

            mel_loss = mel_loss * self.lambda_mel
            bw_kl_loss = bw_kl_loss * self.lambda_bw_kl
            fw_kl_loss = fw_kl_loss * self.lambda_fw_kl
            dur_loss = dur_loss * self.lambda_dur
            adv_loss = adv_loss * self.lambda_adv
            adv_loss_e2e = adv_loss_e2e * self.lambda_adv_e2e
            fm_loss = fm_loss * self.lambda_feat_match
            loss = mel_loss + bw_kl_loss + fw_kl_loss + dur_loss + adv_loss + adv_loss_e2e + fm_loss

            stats = dict(
                gen_loss=loss.item(),
                gen_mel_loss=mel_loss.item(),
                gen_fw_kl_loss=fw_kl_loss.item(),
                gen_bw_kl_loss=bw_kl_loss.item(),
                gen_dur_loss=dur_loss.item(),
                gen_adv_loss=adv_loss.item(),
                gen_adv_e2e_loss=adv_loss_e2e.item(),
                gen_fm_loss=fm_loss.item()
            )
            if self.plot_pred_mos:
                # Caltulate predicted MOS from generated speech waveform.
                with torch.no_grad():
                    # speech_hat_: (B, 1, T)
                    pmos = self.predictor(pre_wav_post.squeeze(1), self.fs).mean()
                stats["generator_predicted_mos"] = pmos.item()

            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

            # reset cache
            if reuse_cache or not self.training:
                self._cache = None

            return {
                "loss": loss,
                "stats": stats,
                "weight": weight,
                "optim_idx": 0,  # needed for trainer
            }

    def _forward_discrminator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        (pre_wav_post, _, z_start_idxs, _,
         _, _, _, _,
         _, _) = outs
        wav_post_segs = get_segments(x=speech,
                                     start_idxs=z_start_idxs * self.generator.upsample_factor,
                                     segment_size=self.generator.segment_size * self.generator.upsample_factor)

        p_hat = self.discriminator(pre_wav_post.detach())
        p = self.discriminator(wav_post_segs)
        with autocast(enabled=False):
            real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
            loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            discriminator_real_loss=real_loss.item(),
            discriminator_fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            feats (Tensor): Feature tensor (T_feats, aux_channels).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            durations (Tensor): Ground-truth duration tensor (T_text,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * att_w (Tensor): Monotonic attention weight tensor (T_feats, T_text).
                * duration (Tensor): Predicted duration tensor (T_text,).

        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        if sids is not None:
            sids = sids.view(1)
        if lids is not None:
            lids = lids.view(1)
        if durations is not None:
            durations = durations.view(1, 1, -1)

        # inference
        if use_teacher_forcing:
            assert feats is not None
            feats = feats[None].transpose(1, 2)
            feats_lengths = torch.tensor(
                [feats.size(2)],
                dtype=torch.long,
                device=feats.device,
            )
            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                max_len=max_len,
                use_teacher_forcing=use_teacher_forcing,
            )
        else:
            wav, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                dur=durations,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
            )
        return dict(wav=wav.view(-1), duration=dur[0])