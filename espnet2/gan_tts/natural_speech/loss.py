# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS-related loss modules.

This code is based on https://github.com/jaywalnut310/vits.

"""
from espnet2.gan_tts.natural_speech.soft_dtw import SoftDTW
from torch import nn
import torch
import torch.distributions as D


class KLDivergenceLoss(torch.nn.Module):
    """KL divergence loss."""

    def forward(
        self,
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss.

        Args:
            z_p (Tensor): Flow hidden representation (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
            z_mask (Tensor): Mask tensor (B, 1, T_feats).

        Returns:
            Tensor: KL divergence loss.

        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        loss = kl / torch.sum(z_mask)

        return loss


class KLDivergenceLossWithoutFlow(torch.nn.Module):
    """KL divergence loss without flow."""

    def forward(
        self,
        m_q: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss without flow.

        Args:
            m_q (Tensor): Posterior encoder projected mean (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
        """
        posterior_norm = D.Normal(m_q, torch.exp(logs_q))
        prior_norm = D.Normal(m_p, torch.exp(logs_p))
        loss = D.kl_divergence(posterior_norm, prior_norm).mean()
        return loss


class SoftDTWKLLoss(nn.Module):
    def __init__(self,
                 use_cuda: bool,
                 gamma: float = 1.0,
                 normalize: bool = False,
                 bandwidth: float = None,
                 warp: float = None
                 ) -> None:
        super().__init__()
        self.soft_dtw = SoftDTW(use_cuda=use_cuda,
                                gamma=gamma,
                                normalize=normalize,
                                bandwidth=bandwidth,
                                warp=warp)

    @staticmethod
    def get_sdtw_kl_matrix(z_p: torch.Tensor,
                           logs_q: torch.Tensor,
                           m_p: torch.Tensor,
                           logs_p: torch.Tensor):
        """
        returns kl matrix with shape [b, t_tp, t_tq]
        z_p, logs_q: [b, h, t_tq]
        m_p, logs_p: [b, h, t_tp]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()

        t_tp, t_tq = m_p.size(-1), z_p.size(-1)
        b, h, t_tp = m_p.shape

        kls = torch.zeros((b, t_tp, t_tq), dtype=z_p.dtype, device=z_p.device)
        for i in range(h):
            logs_p_, m_p_, logs_q_, z_p_ = (
                logs_p[:, i, :, None],
                m_p[:, i, :, None],
                logs_q[:, i, None, :],
                z_p[:, i, None, :],
            )
            kl = logs_p_ - logs_q_ - 0.5  # [b, t_tp, t_tq]
            kl += 0.5 * ((z_p_ - m_p_) ** 2) * torch.exp(-2.0 * logs_p_)
            kls += kl
        return kls

    def forward(self, z_p, logs_q, m_p, logs_p, p_mask, q_mask):
        INF = 1e5
        kl = self.get_sdtw_kl_matrix(z_p, logs_q, m_p, logs_p)  # [b t_tp t_tq]
        kl = torch.nn.functional.pad(kl, (0, 1, 0, 1), "constant", 0)
        p_mask = torch.nn.functional.pad(p_mask, (0, 1), "constant", 0)
        q_mask = torch.nn.functional.pad(q_mask, (0, 1), "constant", 0)

        kl.masked_fill_(p_mask[:, :, None].bool() ^ q_mask[:, None, :].bool(), INF)
        kl.masked_fill_((~p_mask[:, :, None].bool()) & (~q_mask[:, None, :].bool()), 0)
        res = self.soft_dtw(kl).sum() / p_mask.sum()
        return res
