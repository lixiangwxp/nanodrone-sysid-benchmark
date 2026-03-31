import torch
import torch.nn as nn

from models.models import PhysQuadModel, ResidualQuadModel


class MotorLagLayer(nn.Module):
    """First-order actuator lag on motor angular speeds."""

    def __init__(self, lag_mode="per_motor", alpha_init=0.85):
        super().__init__()
        if lag_mode not in {"shared", "per_motor"}:
            raise ValueError("lag_mode must be one of {'shared', 'per_motor'}")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1)")

        self.lag_mode = lag_mode
        param_shape = (1,) if lag_mode == "shared" else (4,)
        alpha = torch.full(param_shape, float(alpha_init), dtype=torch.float32)
        self.logit_alpha = nn.Parameter(torch.logit(alpha))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def forward(self, u_eff_prev, u_raw):
        alpha = self.alpha.view(*([1] * (u_raw.ndim - 1)), -1)
        return alpha * u_eff_prev + (1.0 - alpha) * u_raw


class LagPhysResQuadModel(nn.Module):
    """
    Physics + residual model with actuator lag memory.

    Inputs are expected in normalized space:
      x_norm: [pos(3), vel(3), so3_log(3), omega(3)]
      u_norm: raw motor angular speeds
    """

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        use_aux_head=False,
        aux_dim=1,
    ):
        super().__init__()
        if not isinstance(phys, PhysQuadModel):
            raise TypeError("phys must be an instance of PhysQuadModel")
        if not isinstance(residual, ResidualQuadModel):
            raise TypeError("residual must be an instance of ResidualQuadModel")

        self.phys = phys
        self.residual = residual
        self.dt = phys.dt
        self.use_aux_head = use_aux_head
        self.aux_dim = aux_dim
        self.lag_layer = MotorLagLayer(lag_mode=lag_mode, alpha_init=alpha_init)

        state_dim = residual.out.out_features
        control_dim = 4
        self._validate_residual_input(state_dim=state_dim, control_feat_dim=12)

        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(state_dim + 12, 64),
                nn.ReLU(),
                nn.Linear(64, aux_dim),
            )
        else:
            self.aux_head = None

    @staticmethod
    def _scaler_to_tensors(scaler, dim):
        if scaler is None:
            mean = torch.zeros(dim, dtype=torch.float32)
            scale = torch.ones(dim, dtype=torch.float32)
            return mean, scale

        mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)
        return mean, scale

    def _validate_residual_input(self, state_dim, control_feat_dim):
        first_linear = None
        for layer in self.residual.mlp:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break

        if first_linear is None:
            raise ValueError("Residual model MLP must contain at least one Linear layer")

        expected_in_dim = state_dim + control_feat_dim
        if first_linear.in_features != expected_in_dim:
            raise ValueError(
                "ResidualQuadModel input dimension mismatch: "
                f"expected first layer input {expected_in_dim}, "
                f"got {first_linear.in_features}."
            )

    @property
    def alpha(self):
        return self.lag_layer.alpha

    def x_denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real):
        return (x_real - self.x_mean) / self.x_scale

    def u_denorm(self, u_norm):
        return u_norm * self.u_scale + self.u_mean

    def u_normed(self, u_real):
        return (u_real - self.u_mean) / self.u_scale

    def motor_to_phys_diff(self, u_mot):
        """Differentiable motor-speed to normalized thrust / torque map."""
        omega2 = u_mot ** 2
        T = self.phys.Kt * omega2.sum(dim=1)
        tau_x = self.phys.Kt * self.phys.arm * (
            (omega2[:, 2] + omega2[:, 3]) - (omega2[:, 0] + omega2[:, 1])
        )
        tau_y = self.phys.Kt * self.phys.arm * (
            (omega2[:, 1] + omega2[:, 2]) - (omega2[:, 0] + omega2[:, 3])
        )
        tau_z = self.phys.Kc * (
            (omega2[:, 0] + omega2[:, 2]) - (omega2[:, 1] + omega2[:, 3])
        )

        T_norm = T / self.phys.T_max
        tau_norm = torch.stack([tau_x, tau_y, tau_z], dim=1) / self.phys.max_torque
        return torch.cat([T_norm.unsqueeze(1), tau_norm], dim=1)

    def physics_step_from_motors(self, x_real, u_eff_real):
        """Run one physics step from motor speeds while staying in the 12D state space."""
        pos = x_real[:, 0:3]
        vel = x_real[:, 3:6]
        so3 = x_real[:, 6:9]
        omega = x_real[:, 9:12]

        quat = self.phys.so3_log_to_quat(so3)
        x_quat = torch.cat([pos, vel, quat, omega], dim=-1)
        u_phys = self.motor_to_phys_diff(u_eff_real)

        x_next_quat = self.phys._step_from_phys(x_quat, u_phys)
        pos_next = x_next_quat[:, 0:3]
        vel_next = x_next_quat[:, 3:6]
        quat_next = x_next_quat[:, 6:10]
        omega_next = x_next_quat[:, 10:13]
        so3_next = self.phys.quat_to_so3_log(quat_next)

        return torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)

    def forward(self, x0, u_seq, return_aux=False):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        if x0.ndim == 3:
            x_norm = x0.squeeze(1)
        else:
            x_norm = x0

        if x_norm.ndim != 2:
            raise ValueError("x0 must have shape (B,12) or (B,1,12)")
        if u_seq.ndim != 3 or u_seq.shape[-1] != 4:
            raise ValueError("u_seq must have shape (B,T,4) or (B,4)")

        _, horizon, _ = u_seq.shape
        preds = []
        aux_preds = [] if (return_aux and self.use_aux_head) else None

        u_eff_prev_real = self.u_denorm(u_seq[:, 0, :])

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)
            u_eff_real = self.lag_layer(u_eff_prev_real, u_raw_real)

            x_real = self.x_denorm(x_norm)
            # Keep the physics rollout fixed while preserving gradients through lag -> residual.
            with torch.no_grad():
                x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            u_eff_norm = self.u_normed(u_eff_real)
            feat_u = torch.cat(
                [u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm],
                dim=-1,
            )

            residual_in = torch.cat([x_norm, feat_u], dim=-1)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))

            if aux_preds is not None:
                aux_preds.append(self.aux_head(residual_in).unsqueeze(1))

            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        pred_seq = torch.cat(preds, dim=1)

        if not return_aux:
            return pred_seq

        if aux_preds is None:
            return pred_seq, None

        aux_pred_seq = torch.cat(aux_preds, dim=1)
        return pred_seq, aux_pred_seq


class LagPhysResGRUModel(LagPhysResQuadModel):
    """Physics + residual model with GRU-conditioned dynamic actuator lag."""

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        hidden_dim=64,
    ):
        nn.Module.__init__(self)
        if not isinstance(phys, PhysQuadModel):
            raise TypeError("phys must be an instance of PhysQuadModel")
        if not isinstance(residual, ResidualQuadModel):
            raise TypeError("residual must be an instance of ResidualQuadModel")

        self.phys = phys
        self.residual = residual
        self.dt = phys.dt
        self.use_aux_head = False
        self.aux_dim = 0
        self.aux_head = None
        self.lag_layer = MotorLagLayer(lag_mode=lag_mode, alpha_init=alpha_init)
        self.gru_hidden_dim = hidden_dim

        state_dim = residual.out.out_features
        control_dim = 4
        feature_dim = state_dim + 3 * control_dim + hidden_dim
        self._validate_residual_input(
            state_dim=state_dim,
            control_feat_dim=3 * control_dim + hidden_dim,
        )

        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

        self.h_init = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.alpha_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gru_cell = nn.GRUCell(feature_dim, hidden_dim)

    @staticmethod
    def _pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h):
        return torch.cat(
            [x_norm, u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm, h],
            dim=-1,
        )

    def forward(self, x0, u_seq, return_force=False):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        if x0.ndim == 3:
            x_norm = x0.squeeze(1)
        else:
            x_norm = x0

        if x_norm.ndim != 2:
            raise ValueError("x0 must have shape (B,12) or (B,1,12)")
        if u_seq.ndim != 3 or u_seq.shape[-1] != 4:
            raise ValueError("u_seq must have shape (B,T,4) or (B,4)")

        _, horizon, _ = u_seq.shape
        preds = []
        h = self.h_init(x_norm)
        u_eff_prev_real = self.u_denorm(u_seq[:, 0, :])

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)

            u_eff_seed_real = self.lag_layer(u_eff_prev_real, u_raw_real)
            u_eff_seed_norm = self.u_normed(u_eff_seed_real)
            alpha_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_seed_norm, h)
            alpha_t = torch.sigmoid(self.alpha_head(alpha_in))

            u_eff_real = alpha_t * u_eff_prev_real + (1.0 - alpha_t) * u_raw_real
            u_eff_norm = self.u_normed(u_eff_real)

            x_real = self.x_denorm(x_norm)
            with torch.no_grad():
                x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            gru_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            h = self.gru_cell(gru_in, h)

            residual_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        return torch.cat(preds, dim=1)


class LagPhysResGRUForceModel(LagPhysResGRUModel):
    """GRU-conditioned lag model with learned initial lag state and residual force."""

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        hidden_dim=64,
    ):
        super().__init__(
            phys=phys,
            residual=residual,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=hidden_dim,
        )

        state_dim = residual.out.out_features
        control_dim = 4
        self.u_init_head = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim),
        )
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        nn.init.zeros_(self.u_init_head[-1].weight)
        nn.init.zeros_(self.u_init_head[-1].bias)
        nn.init.zeros_(self.force_head[-1].weight)
        nn.init.zeros_(self.force_head[-1].bias)

    def forward(self, x0, u_seq):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        if x0.ndim == 3:
            x_norm = x0.squeeze(1)
        else:
            x_norm = x0

        if x_norm.ndim != 2:
            raise ValueError("x0 must have shape (B,12) or (B,1,12)")
        if u_seq.ndim != 3 or u_seq.shape[-1] != 4:
            raise ValueError("u_seq must have shape (B,T,4) or (B,4)")

        _, horizon, _ = u_seq.shape
        preds = []
        force_preds = []
        u_eff_seq = []
        h = self.h_init(x_norm)

        u0_norm = u_seq[:, 0, :]
        u0_real = self.u_denorm(u0_norm)
        delta_u0 = self.u_init_head(torch.cat([x_norm, u0_norm], dim=-1))
        u_eff_prev_real = u0_real + delta_u0

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)

            u_eff_seed_real = self.lag_layer(u_eff_prev_real, u_raw_real)
            u_eff_seed_norm = self.u_normed(u_eff_seed_real)
            alpha_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_seed_norm, h)
            alpha_t = torch.sigmoid(self.alpha_head(alpha_in))

            u_eff_real = alpha_t * u_eff_prev_real + (1.0 - alpha_t) * u_raw_real
            u_eff_norm = self.u_normed(u_eff_real)

            x_real = self.x_denorm(x_norm)
            with torch.no_grad():
                x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)

            gru_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            h = self.gru_cell(gru_in, h)

            delta_f_b = self.force_head(h)
            x_force_next_real = self.phys.apply_force(
                x_real,
                u_eff_real,
                delta_f_b,
                x_phys_next_real=x_phys_next_real,
            )

            residual_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            dx_res_norm = self.residual.out(self.residual.mlp(residual_in))
            dx_res_real = dx_res_norm * self.x_scale
            x_next_real = x_force_next_real + dx_res_real
            x_force_next_norm = self.x_normed(x_force_next_real)
            x_next_norm = self.x_normed(x_next_real)

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_force_next_norm)

            preds.append(x_next_norm.unsqueeze(1))
            force_preds.append(delta_f_b.unsqueeze(1))
            u_eff_seq.append(u_eff_real.unsqueeze(1))
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        pred_seq = torch.cat(preds, dim=1)
        if not return_force:
            return pred_seq

        force_pred_seq = torch.cat(force_preds, dim=1)
        u_eff_seq_real = torch.cat(u_eff_seq, dim=1)
        return pred_seq, force_pred_seq, u_eff_seq_real


__all__ = [
    "MotorLagLayer",
    "LagPhysResQuadModel",
    "LagPhysResGRUModel",
    "LagPhysResGRUForceModel",
]
