import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class LossConfig:
    temporal_loss: str
    use_geo: bool
    use_aux: bool
    use_force: bool
    beta_geo: float
    beta_aux: float
    beta_force: float
    w_rot: float
    w_omega: float
    lambda_mse: float = 0.1
    aux_loss_type: str = "smooth_l1"


def build_loss_config(
    loss_type,
    beta_geo,
    beta_aux,
    beta_force,
    w_rot,
    w_omega,
    use_geo=False,
    use_aux=False,
    use_force=False,
    lambda_mse=0.1,
    aux_loss_type="smooth_l1",
):
    return LossConfig(
        temporal_loss=loss_type,
        use_geo=use_geo,
        use_aux=use_aux,
        use_force=use_force,
        beta_geo=beta_geo,
        beta_aux=beta_aux,
        beta_force=beta_force,
        w_rot=w_rot,
        w_omega=w_omega,
        lambda_mse=lambda_mse,
        aux_loss_type=aux_loss_type,
    )


def compute_mixed_temporal_loss(pred_seq, true_seq, profile="mixed"):
    window = min(10, pred_seq.shape[1])
    loss_exp = WeightedMSELoss(lambda_=0.03)(pred_seq, true_seq)
    loss_uniform = torch.mean((pred_seq - true_seq) ** 2)
    loss_tail = torch.mean((pred_seq[:, -window:] - true_seq[:, -window:]) ** 2)

    if profile == "mixed":
        loss_early = loss_exp.detach().new_tensor(0.0)
        total_loss = 0.5 * loss_exp + 0.3 * loss_uniform + 0.2 * loss_tail
    elif profile == "mixed_early":
        loss_early = torch.mean((pred_seq[:, :window] - true_seq[:, :window]) ** 2)
        total_loss = (
            0.35 * loss_exp
            + 0.25 * loss_uniform
            + 0.20 * loss_tail
            + 0.20 * loss_early
        )
    else:
        raise ValueError(f"Unsupported mixed temporal loss profile: {profile}")

    zero = total_loss.detach().new_tensor(0.0)
    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_mse": total_loss.detach(),
        "loss_geo": zero,
        "loss_aux": zero,
        "loss_exp": loss_exp.detach(),
        "loss_uniform": loss_uniform.detach(),
        "loss_tail": loss_tail.detach(),
        "loss_early": loss_early.detach(),
    }
    return total_loss, loss_dict


class AblationCriterion(nn.Module):
    def __init__(self, loss_config, x_scaler):
        super().__init__()
        self.loss_config = loss_config
        self.temporal_exp_loss = WeightedMSELoss(lambda_=loss_config.lambda_mse)
        self.geo_loss = None
        self.aux_loss = None

        if loss_config.use_geo:
            if x_scaler is None:
                raise ValueError("x_scaler is required when use_geo=True")
            self.register_buffer("x_mean", torch.as_tensor(x_scaler.mean_, dtype=torch.float32))
            self.register_buffer("x_scale", torch.as_tensor(x_scaler.scale_, dtype=torch.float32))
            self.geo_loss = WeightedGeodesicLoss(
                lambda_=loss_config.lambda_mse,
                w_pos=0.0,
                w_vel=0.0,
                w_rot=loss_config.w_rot,
                w_omega=loss_config.w_omega,
            )

        if loss_config.use_aux:
            self.aux_loss = CompositeAblationLoss._build_aux_loss(loss_config.aux_loss_type)

    def denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    def _compute_temporal_loss(self, pred_seq_norm, true_seq_norm):
        if self.loss_config.temporal_loss == "exp":
            total_loss = self.temporal_exp_loss(pred_seq_norm, true_seq_norm)
            zero = total_loss.detach().new_tensor(0.0)
            loss_dict = {
                "loss_total": total_loss.detach(),
                "loss_mse": total_loss.detach(),
                "loss_geo": zero,
                "loss_aux": zero,
            }
            return total_loss, loss_dict

        if self.loss_config.temporal_loss in {"mixed", "mixed_early"}:
            return compute_mixed_temporal_loss(
                pred_seq_norm,
                true_seq_norm,
                profile=self.loss_config.temporal_loss,
            )

        raise ValueError(
            f"Unsupported temporal loss profile: {self.loss_config.temporal_loss}"
        )

    def forward(self, pred_seq_norm, true_seq_norm, aux_pred=None, aux_true=None):
        temporal_loss, loss_dict = self._compute_temporal_loss(pred_seq_norm, true_seq_norm)
        loss_geo = pred_seq_norm.new_tensor(0.0)
        loss_aux = pred_seq_norm.new_tensor(0.0)

        numerator = temporal_loss
        denominator = 1.0

        if self.loss_config.use_geo:
            pred_seq_real = self.denorm(pred_seq_norm)
            true_seq_real = self.denorm(true_seq_norm)
            loss_geo = self.geo_loss(pred_seq_real, true_seq_real)
            numerator = numerator + self.loss_config.beta_geo * loss_geo
            denominator += self.loss_config.beta_geo

        if self.loss_config.use_aux:
            if aux_pred is None or aux_true is None:
                raise ValueError("aux_pred and aux_true are required when use_aux=True")
            loss_aux = self.aux_loss(aux_pred, aux_true)
            numerator = numerator + self.loss_config.beta_aux * loss_aux
            denominator += self.loss_config.beta_aux

        total_loss = numerator / denominator
        loss_dict["loss_total"] = total_loss.detach()
        loss_dict["loss_mse"] = temporal_loss.detach()
        loss_dict["loss_geo"] = loss_geo.detach()
        loss_dict["loss_aux"] = loss_aux.detach()
        return total_loss, loss_dict


def build_criterion(loss_config, x_scaler):
    return AblationCriterion(loss_config=loss_config, x_scaler=x_scaler)


def _compute_force_loss(model, batch, force_pred_seq, u_eff_seq_real, device, beta_force):
    if len(batch) < 4:
        zero = force_pred_seq.detach().new_tensor(0.0)
        return zero, zero

    # IO-only optimization: allow pinned-memory batches to overlap host->device copies.
    aux_seq = batch[3].to(device, non_blocking=True)
    if aux_seq.shape[-1] < 3:
        zero = force_pred_seq.detach().new_tensor(0.0)
        return zero, zero

    f_meas_body = model.phys.m * aux_seq[..., :3]
    thrust = model.phys.Kt * (u_eff_seq_real ** 2).sum(dim=-1, keepdim=True)
    zeros = torch.zeros_like(thrust)
    f_phys_body = torch.cat([zeros, zeros, thrust], dim=-1)
    force_target_seq = (f_meas_body - f_phys_body).detach()

    force_loss = torch.nn.functional.smooth_l1_loss(force_pred_seq, force_target_seq)
    weighted_force_loss = beta_force * force_loss
    return weighted_force_loss, force_loss.detach()


def compute_loss(model, criterion, batch, device, loss_config):
    x0, u_seq, true_seq = batch[:3]
    # IO-only optimization: preserve values while enabling async H2D copies when available.
    x0 = x0.to(device, non_blocking=True)
    u_seq = u_seq.to(device, non_blocking=True)
    true_seq = true_seq.to(device, non_blocking=True)
    x_hist = batch[3] if len(batch) >= 5 else None
    u_hist = batch[4] if len(batch) >= 5 else None
    if x_hist is not None:
        x_hist = x_hist.to(device, non_blocking=True)
        u_hist = u_hist.to(device, non_blocking=True)

    aux_seq = (
        batch[3].to(device, non_blocking=True)
        if len(batch) == 4
        else None
    )
    needs_history = bool(
        getattr(model, "use_hist_init", False)
        or getattr(model, "actbank_use_history", False)
        or getattr(model, "requires_history", False)
    )
    if needs_history and x_hist is None:
        raise ValueError("Model requires history but batch does not provide x_hist/u_hist")

    aux_pred = None
    force_pred_seq = None
    u_eff_seq_real = None

    if loss_config.use_force:
        pred_seq, force_pred_seq, u_eff_seq_real = model(x0, u_seq, return_force=True)
    elif loss_config.use_aux:
        pred_seq, aux_pred = model(x0, u_seq, return_aux=True)
    else:
        if needs_history:
            pred_seq = model(x0, u_seq, x_hist=x_hist, u_hist=u_hist)
        else:
            pred_seq = model(x0, u_seq)

    total_loss, loss_dict = criterion(
        pred_seq,
        true_seq,
        aux_pred=aux_pred,
        aux_true=aux_seq,
    )

    if loss_config.use_force:
        weighted_force_loss, force_loss_value = _compute_force_loss(
            model=model,
            batch=batch,
            force_pred_seq=force_pred_seq,
            u_eff_seq_real=u_eff_seq_real,
            device=device,
            beta_force=loss_config.beta_force,
        )
        total_loss = total_loss + weighted_force_loss
        loss_dict["loss_total"] = total_loss.detach()
        loss_dict["loss_force"] = force_loss_value

    return total_loss, loss_dict


__all__ = [
    "DenormRotGeodesicLoss",
    "CompositeAblationLoss",
    "LossConfig",
    "build_loss_config",
    "compute_mixed_temporal_loss",
    "AblationCriterion",
    "build_criterion",
    "compute_loss",
]
