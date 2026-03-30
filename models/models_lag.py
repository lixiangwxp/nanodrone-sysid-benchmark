"""
带执行器滞后的物理 + 残差四旋翼模型。

本模块在 PhysQuadModel + ResidualQuadModel 基础上，对电机角速度引入一阶低通（滞后），
使「有效电机指令」随时间平滑跟踪原始指令；状态/控制在训练管道中仍以 StandardScaler
的均值方差做归一化，与主仓库 dataset 一致。
"""
import torch
import torch.nn as nn

from models.models import PhysQuadModel, ResidualQuadModel


class MotorLagLayer(nn.Module):
    """First-order actuator lag on motor angular speeds."""

    def __init__(self, lag_mode="per_motor", alpha_init=0.85):
        super().__init__()
        # lag_mode: "shared" 四个电机共用一个 alpha；"per_motor" 每个电机独立 alpha
        if lag_mode not in {"shared", "per_motor"}:
            raise ValueError("lag_mode must be one of {'shared', 'per_motor'}")
        # alpha 为 IIR 低通系数，必须在 (0,1)：越大越「粘」在上一时刻有效值上
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1)")

        self.lag_mode = lag_mode
        # 可学习参数形状：(1,) 或 (4,)；训练时用 logit 参数化保证 sigmoid 后落在 (0,1)
        param_shape = (1,) if lag_mode == "shared" else (4,)
        alpha = torch.full(param_shape, float(alpha_init), dtype=torch.float32)
        self.logit_alpha = nn.Parameter(torch.logit(alpha))
        #真正参与前向、且需落在 (0,1) 的，是属性里的 self.alpha = torch.sigmoid(self.logit_alpha)（见 MotorLagLayer.alpha）。
        #优化器改的是 logit_alpha（无界实数）

    @property
    def alpha(self):
        # 将无约束的 logit 压回 (0,1)，得到每步混合系数
        return torch.sigmoid(self.logit_alpha)

    def forward(self, u_eff_prev, u_raw):
        # u_eff_prev, u_raw: 均为物理空间电机角速度 (B,4) 或与 batch 维对齐的扩展形状
        # 一阶滞后：u_eff = alpha * u_eff_prev + (1-alpha) * u_raw（按元素或按电机）
        alpha = self.alpha.view(*([1] * (u_raw.ndim - 1)), -1)
        #把self.alpha这个一维向量，重塑成和u_raw维度一样多的张量，前面所有维度都填1，最后一维保留原始数据
        #不改变 α 的数值，只改形状，以便广播乘法。
        return alpha * u_eff_prev + (1.0 - alpha) * u_raw #(B, 4)


class LagPhysResQuadModel(nn.Module):
    """
    Physics + residual model with actuator lag memory.
    Inputs are expected in normalized space:
      x_norm: [pos(3), vel(3), so3_log(3), omega(3)]
      u_norm: raw motor angular speeds
    """
    def __init__(
        self,
        phys,#提供刚体 + 旋翼的物理一步积分
        residual,#加在物理预测上：x_next = x_phys_next + dx_res。
        x_scaler,#把 DataLoader 给的 归一化状态 x_norm 变回物理量
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,#滞后系数 初始值（必须在 (0,1)，
        use_aux_head=False,#要不要挂一个 辅助预测头。
        aux_dim=1,#辅助头 输出维度
    ):
        super().__init__()
        if not isinstance(phys, PhysQuadModel):
            raise TypeError("phys must be an instance of PhysQuadModel")
        if not isinstance(residual, ResidualQuadModel):
            raise TypeError("residual must be an instance of ResidualQuadModel")

        #把物理模型、残差网络挂到 self 上：nn.Module 会把赋值给 self.xxx 且也是 Module 的对象登记为子模块（phys、residual、lag_layer 都会在参数遍历里出现
        self.phys = phys
        self.residual = residual
        self.dt = phys.dt
        self.use_aux_head = use_aux_head
        self.aux_dim = aux_dim
        self.lag_layer = MotorLagLayer(lag_mode=lag_mode, alpha_init=alpha_init)

        # residual.out 的最后一层线性输出维度 = 状态残差维度（通常 12）
        state_dim = residual.out.out_features
        control_dim = 4 #四个电机
        # 残差 MLP 第一项输入必须是 x_norm(12) + feat_u(12)，此处固定校验为 12+12=24
        self._validate_residual_input(state_dim=state_dim, control_feat_dim=12)

        # 缓存 sklearn StandardScaler 的 mean_/scale_，用于在模型内做 x/u 的反归一与归一
        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

        # 可选辅助头：在 residual 同一输入上预测 aux_dim 维标量/向量（如侧任务）
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(state_dim + 12, 64),
                nn.ReLU(),
                nn.Linear(64, aux_dim),
            )
        else:
            self.aux_head = None

    @staticmethod
    #@staticmethod 是啥？表示不依赖self/cls，逻辑只跟传入参数有关。这里放在类里只是为了命名空间整齐
    def _scaler_to_tensors(scaler, dim):
        # 无 scaler 时假定数据已为「标准空间」，不归一化
        if scaler is None:
            mean = torch.zeros(dim, dtype=torch.float32)
            scale = torch.ones(dim, dtype=torch.float32)
            return mean, scale

        mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)
        return mean, scale
    '''residual_in 长什么样
    x_norm:当前归一化状态，维度 = state_dim(一般是 12 = pos+vel+so3+omega)。
    feat_u:由三截各 4 维拼成 = u_raw_norm ∥ u_eff_norm ∥ (u_raw_norm - u_eff_norm)，共 4+4+4 = 12 维。
    所以 residual_in 的总维度 = state_dim + 12。'''
    '''这是在核对 「残差 MLP 吃进去的向量长度」 是否等于 x_norm + 12 维电机特征'''
    def _validate_residual_input(self, state_dim, control_feat_dim):
        # feat_u 设计为 12 维：见 forward 中 u_raw, u_eff, u_raw-u_eff 的拼接
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
                "ResidualQuadModel must be instantiated with input_dim=12 for "
                "LagPhysResQuadModel."
            )

    @property
    def alpha(self):
        return self.lag_layer.alpha

    def x_denorm(self, x_norm):
        # x_real = x_norm * scale + mean（与 sklearn inverse_transform 一致）
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real):
        return (x_real - self.x_mean) / self.x_scale

    def u_denorm(self, u_norm):
        return u_norm * self.u_scale + self.u_mean

    def u_normed(self, u_real):
        return (u_real - self.u_mean) / self.u_scale

    def motor_to_phys_diff(self, u_mot):
        """Differentiable motor-speed to normalized thrust / torque map."""
        # 与 PhysQuadModel.motor_to_phys 相同的 Kt/Kc/arm 二次力矩模型，但保留梯度供反传
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
        # 外状态：pos, vel, so3_log, omega；内部一步推进仍用 phys 的四元数接口
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
        # 统一成 (B,T,4) 与 (B,12) 的归一化状态，便于多步循环
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

        # 滞后初值：用第一拍的原始指令（反归一后）作为上一时刻有效电机速，与循环内第一拍衔接
        u_eff_prev_real = self.u_denorm(u_seq[:, 0, :])

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)
            # 有效执行器输入：低通后的电机角速度（实值 rad/s）
            u_eff_real = self.lag_layer(u_eff_prev_real, u_raw_real)

            # 物理分支：实值状态 -> 物理一步 -> 再压回归一化空间
            x_real = self.x_denorm(x_norm)
            x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            # 残差分支：构造 12 维控制特征（指令、有效、差分），与当前 x_norm 拼接
            u_eff_norm = self.u_normed(u_eff_real)
            feat_u = torch.cat(
                [u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm],
                dim=-1,
            )

            residual_in = torch.cat([x_norm, feat_u], dim=-1)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            # 数值护栏：若物理+残差出现 inf/nan，回退为纯物理预测，避免整条序列污染
            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))

            if aux_preds is not None:
                aux_preds.append(self.aux_head(residual_in).unsqueeze(1))

            # 自回归：下一步的「当前归一化状态」与滞后记忆
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        pred_seq = torch.cat(preds, dim=1)

        if not return_aux:
            return pred_seq

        if aux_preds is None:
            return pred_seq, None

        aux_pred_seq = torch.cat(aux_preds, dim=1)
        return pred_seq, aux_pred_seq


__all__ = ["MotorLagLayer", "LagPhysResQuadModel"]
