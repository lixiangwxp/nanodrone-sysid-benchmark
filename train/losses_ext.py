import math

import torch
import torch.nn as nn
from pytorch3d.transforms import so3_exp_map, so3_relative_angle

from train.losses import WeightedGeodesicLoss, WeightedMSELoss


def clamp_rotvec(phi, max_angle=math.inf, eps=1e-8):
    """Clamp rotation vectors to a safe range for SO(3) numerics."""
    angle = torch.norm(phi, dim=-1, keepdim=True)
    scale = torch.clamp(angle, max=max_angle) / (angle + eps)
    return phi * scale


class DenormRotGeodesicLoss(nn.Module):
    """
    SO(3) geodesic loss computed on denormalized rotation vectors only.

    pred_norm and true_norm must have shape (B, N, 12).
    """

    def __init__(self, x_scaler, lambda_=0.1, clamp_rot=True):
        super().__init__()
        if x_scaler is None:
            raise ValueError("x_scaler is required for DenormRotGeodesicLoss")

        self.lambda_ = lambda_
        self.clamp_rot = clamp_rot

        self.register_buffer(
            "x_mean", torch.as_tensor(x_scaler.mean_, dtype=torch.float32)
        )
        self.register_buffer(
            "x_scale", torch.as_tensor(x_scaler.scale_, dtype=torch.float32)
        )

    def denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    def forward(self, pred_norm, true_norm):
        if pred_norm.shape != true_norm.shape:
            raise ValueError("pred_norm and true_norm must have the same shape")
        if pred_norm.ndim != 3 or pred_norm.shape[-1] != 12:
            raise ValueError("Expected inputs with shape (B, N, 12)")

        pred_real = self.denorm(pred_norm)
        true_real = self.denorm(true_norm)

        r_pred = pred_real[..., 6:9]
        r_true = true_real[..., 6:9]

        if self.clamp_rot:
            r_pred = clamp_rotvec(r_pred)
            r_true = clamp_rotvec(r_true)

        batch_size, horizon, _ = r_pred.shape
        r_pred_flat = r_pred.reshape(batch_size * horizon, 3)
        r_true_flat = r_true.reshape(batch_size * horizon, 3)

        R_pred = so3_exp_map(r_pred_flat)
        R_true = so3_exp_map(r_true_flat)
        rot_err = so3_relative_angle(R_pred, R_true).view(batch_size, horizon)

        weights = torch.exp(-self.lambda_ * torch.arange(horizon, device=pred_norm.device))
        weights = weights / weights.sum()

        return ((rot_err ** 2) * weights.view(1, horizon)).mean()


class CompositeAblationLoss(nn.Module):
    """Weighted MSE with optional denormalized geodesic and auxiliary terms."""

    def __init__(
        self,
        x_scaler,
        lambda_mse=0.1,
        use_geo=False,
        beta_geo=0.01,
        use_aux=False,
        beta_aux=0.05,
        aux_loss_type="smooth_l1",
        w_rot=2.0,
        w_omega=2.0,
    ):
        super().__init__()
        if x_scaler is None:
            raise ValueError("x_scaler is required for CompositeAblationLoss")

        self.use_geo = use_geo
        self.beta_geo = beta_geo
        self.use_aux = use_aux
        self.beta_aux = beta_aux
        self.w_rot = w_rot
        self.w_omega = w_omega

        self.mse_loss = WeightedMSELoss(lambda_=lambda_mse)
        self.register_buffer(
            "x_mean", torch.as_tensor(x_scaler.mean_, dtype=torch.float32)
        )
        self.register_buffer(
            "x_scale", torch.as_tensor(x_scaler.scale_, dtype=torch.float32)
        )
        self.geo_loss = None
        if use_geo:
            self.geo_loss = WeightedGeodesicLoss(
                lambda_=lambda_mse,
                w_pos=0.0,
                w_vel=0.0,
                w_rot=w_rot,
                w_omega=w_omega,
            )
        self.aux_loss = self._build_aux_loss(aux_loss_type) if use_aux else None

    def denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    @staticmethod
    def _build_aux_loss(aux_loss_type):
        if aux_loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        if aux_loss_type == "mse":
            return nn.MSELoss()
        if aux_loss_type == "l1":
            return nn.L1Loss()
        raise ValueError("aux_loss_type must be one of {'smooth_l1', 'mse', 'l1'}")

    def forward(self, pred_seq_norm, true_seq_norm, aux_pred=None, aux_true=None):
        loss_mse = self.mse_loss(pred_seq_norm, true_seq_norm)
        loss_geo = pred_seq_norm.new_tensor(0.0)
        loss_aux = pred_seq_norm.new_tensor(0.0)

        numerator = loss_mse
        denominator = 1.0

        if self.use_geo:
            pred_seq_real = self.denorm(pred_seq_norm)
            true_seq_real = self.denorm(true_seq_norm)
            loss_geo = self.geo_loss(pred_seq_real, true_seq_real)
            numerator = numerator + self.beta_geo * loss_geo
            denominator += self.beta_geo

        if self.use_aux:
            if aux_pred is None or aux_true is None:
                raise ValueError("aux_pred and aux_true are required when use_aux=True")
            loss_aux = self.aux_loss(aux_pred, aux_true)
            numerator = numerator + self.beta_aux * loss_aux
            denominator += self.beta_aux

        total_loss = numerator / denominator

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_mse": loss_mse.detach(),
            "loss_geo": loss_geo.detach(),
            "loss_aux": loss_aux.detach(),
        }
        return total_loss, loss_dict


__all__ = ["DenormRotGeodesicLoss", "CompositeAblationLoss"]
