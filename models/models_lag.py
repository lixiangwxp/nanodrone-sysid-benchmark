import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #真正参与优化器更新的是 logit_alpha，真正用于物理公式的是 alpha。

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)#优化器最擅长更新的是任意实数参数

    def forward(self, u_eff_prev, u_raw):
        alpha = self.alpha.view(*([1] * (u_raw.ndim - 1)), -1)
        return alpha * u_eff_prev + (1.0 - alpha) * u_raw


class ControlTCNEncoder(nn.Module):
    """Encode the full known future control sequence into stepwise and global context."""

    def __init__(self, ctx_dim=32, kernel_size=3, dilations=(1, 2, 4)):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(dilation) for dilation in dilations)

        convs = []
        in_channels = 8
        for dilation in self.dilations:
            convs.append(
                nn.Conv1d(
                    in_channels,
                    self.ctx_dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                )
            )
            in_channels = self.ctx_dim
        self.convs = nn.ModuleList(convs)

        squeeze_dim = max(4, self.ctx_dim // 4)
        self.channel_gate = nn.Sequential(
            nn.Linear(self.ctx_dim, squeeze_dim),
            nn.ReLU(),
            nn.Linear(squeeze_dim, self.ctx_dim),
            nn.Sigmoid(),
        )

    def _causal_conv(self, x, conv):
        left_pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        x = F.pad(x, (left_pad, 0))
        return conv(x)

    def forward(self, u_seq_norm):
        if u_seq_norm.ndim != 3 or u_seq_norm.shape[-1] != 4:
            raise ValueError("u_seq_norm must have shape (B,T,4)")

        du_seq = torch.zeros_like(u_seq_norm)
        du_seq[:, 1:, :] = u_seq_norm[:, 1:, :] - u_seq_norm[:, :-1, :]
        features = torch.cat([u_seq_norm, du_seq], dim=-1).transpose(1, 2)

        hidden = features
        for conv in self.convs:
            hidden = torch.relu(self._causal_conv(hidden, conv))

        channel_gate = self.channel_gate(hidden.mean(dim=-1)).unsqueeze(-1)
        hidden = hidden * channel_gate
        c_seq = hidden.transpose(1, 2)
        c_global = hidden.mean(dim=-1)
        return c_seq, c_global

#在「物理模型 + 残差网络」的基础上，多了一层「电机指令→等效转速」的一阶滞后，并沿时间一步步 roll 出整段预测。
class LagPhysResQuadModel(nn.Module):
    """
    Physics + residual model with actuator lag memory.
    Inputs are expected in normalized space:
      x_norm: [pos(3), vel(3), so3_log(3), omega(3)]
      u_norm: raw motor angular speeds
      外部传进来的 x_norm、u_norm 已是标准化后的张量
      12 维状态拆成：位置 3、速度 3、旋转李代数 3、角速度 3
      u 是 4 路电机角速度(raw 指令)。
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
        for param in self.phys.parameters():
            param.requires_grad_(False)#把物理主干self.phys冻结。

        state_dim = residual.out.out_features#残差网络输出维度
        control_dim = 4#4个电机
        self._validate_residual_input(state_dim=state_dim, control_feat_dim=12)#检查维度对吗


        '''把 sklearn scaler 里的均值和标准差拿出来，转成 PyTorch tensor。'''
        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)

        '''buffer 的意思是：这些张量属于模型，但不是可训练参数。'''
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
    # 处理 scaler
    def _scaler_to_tensors(scaler, dim):
        if scaler is None:
            mean = torch.zeros(dim, dtype=torch.float32)
            scale = torch.ones(dim, dtype=torch.float32)
            return mean, scale

        mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)
        return mean, scale

    #检查residual网络维度
    def _validate_residual_input(self, state_dim, control_feat_dim):
        first_linear = None
        for layer in self.residual.mlp:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break

        expected_in_dim = state_dim + control_feat_dim
        if first_linear.in_features != expected_in_dim:
            raise ValueError(
                "ResidualQuadModel input dimension mismatch: "
                f"expected first layer input {expected_in_dim}, "
                f"got {first_linear.in_features}."
            )

    #暴露 lag 系数
    @property
    def alpha(self):
        return self.lag_layer.alpha

    #归一化/反归一化
    def x_denorm(self, x_norm):
        return x_norm * self.x_scale + self.x_mean
    def x_normed(self, x_real):
        return (x_real - self.x_mean) / self.x_scale
    def u_denorm(self, u_norm):
        return u_norm * self.u_scale + self.u_mean
    def u_normed(self, u_real):
        return (u_real - self.u_mean) / self.u_scale

    
    #电机转速映射到物理输入,变成 physics 可用的推力/力矩输入
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
    


    #输入是12维真实状态x_real和真实电机转速u_eff_real，输出是“经过physics推进一步之后”的下一个 12 维真实状态。
    def physics_step_from_motors(self, x_real, u_eff_real):
        """Run one physics step from motor speeds while staying in the 12D state space.
        p, v, r(so3_log), ω，其中姿态在接口上是 3 维，内部临时换成 quaternion 做积分。

"""
        pos = x_real[:, 0:3]
        vel = x_real[:, 3:6]
        so3 = x_real[:, 6:9]
        omega = x_real[:, 9:12]

        quat = self.phys.so3_log_to_quat(so3)
        x_quat = torch.cat([pos, vel, quat, omega], dim=-1)#(B,13)
        u_phys = self.motor_to_phys_diff(u_eff_real)#4个转速变成力

        x_next_quat = self.phys._step_from_phys(x_quat, u_phys)
        #_step_from_phys 吃的是「机体合力/合力矩」而不是原始 4 个 ω。
        pos_next = x_next_quat[:, 0:3]#_step_from_phys 吃的是「机体合力/合力矩」而不是原始 4 个 ω。
        vel_next = x_next_quat[:, 3:6]
        quat_next = x_next_quat[:, 6:10]#4元数
        omega_next = x_next_quat[:, 10:13]
        so3_next = self.phys.quat_to_so3_log(quat_next)#转回3维

        return torch.cat([pos_next, vel_next, so3_next, omega_next], dim=-1)


    #输入：x0：初始状态，归一化，形状 (B,12) 或 (B,1,12)。
    #u_seq：电机角速度序列，归一化，(B,T,4) 或 (B,4)（后者会先变成单步 (B,1,4)）。
    #输出：pred_seq：(B,T,12)，每一步预测后的归一化状态。
    #若 return_aux=True 且建了 aux_head，还会返回 aux_pred_seq。
    def forward(self, x0, u_seq, return_aux=False):
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)#(B,4) -> (B,1,4)
        if x0.ndim == 3:
            x_norm = x0.squeeze(1)#(B,1,12) -> (B,12)
        else:
            x_norm = x0#(B,12)
        

        #初始化 rollout 容器
        _, horizon, _ = u_seq.shape
        preds = []
        aux_preds = [] if (return_aux and self.use_aux_head) else None
        u_eff_prev_real = self.u_denorm(u_seq[:, 0, :])
        #因为lag模型每一步都需要“上一时刻的有效转速”，所以第一步先拿第一个控制输入反归一化后当 seed。


        #时间循环才是核心
        for t in range(horizon):
            #取当前控制并反归一化
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)
            #等效转速 = 当前转速 + 上一步等效
            u_eff_real = self.lag_layer(u_eff_prev_real, u_raw_real)
            # physics 推进一步
            x_real = self.x_denorm(x_norm)
            x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)
            #构造 residual 网络输入，#输入给 MLP 的特征：x 和 u 的差异。12+12=24
            u_eff_norm = self.u_normed(u_eff_real)
            feat_u = torch.cat(
                [u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm],
                dim=-1,
            )
            residual_in = torch.cat([x_norm, feat_u], dim=-1)

            #残差修正，与状态同维 12，加在物理预测的归一化状态上，补齐模型误差。
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res
            
            #做数值保护并更新下一步输入
            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))

            if aux_preds is not None:
                aux_preds.append(self.aux_head(residual_in).unsqueeze(1))

            x_norm = x_next_norm
         
            u_eff_prev_real = u_eff_real

        #循环结束后返回整段轨迹
        pred_seq = torch.cat(preds, dim=1)

        if not return_aux:
            return pred_seq
        if aux_preds is None:
            return pred_seq, None
        aux_pred_seq = torch.cat(aux_preds, dim=1)
        return pred_seq, aux_pred_seq


class LagPhysResGRUModel(LagPhysResQuadModel):
    """Physics + residual model with GRU-conditioned dynamic actuator lag.
引入一个简单的循环模块和动态滞后机制，而不改变物理模型和残差 MLP 的核心结构。
    通过 GRUCell 维护一个隐藏状态 h_t,模型可以记忆先前步长的信息；
    通过 alpha_head 动态生成滞后系数a,使执行器的惯性更贴合不同状态；
    然后残差 MLP 的输入由 [x_norm, u_raw_norm, u_eff_norm, h_t] 组成，可以利用记忆修正长期偏差。
    """
    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        hidden_dim=64,
        alpha_dim=1,
        use_u_init=False,
        u_init_scale=0.05,
        use_hist_init=False,
        hist_init_scale=0.1,
        use_actbank_alpha=False,
        actbank_use_history=False,
        actbank_taus_ms=(20.0, 50.0, 100.0, 200.0),
        actbank_alpha_scale=0.1,
    ):
        nn.Module.__init__(self)

        # 1) 普通配置 / 非参数属性
        self.phys = phys
        self.residual = residual
        self.dt = phys.dt
        self.use_aux_head = False
        self.aux_dim = 0
        self.aux_head = None

        # 2) 可学习模块：执行器滞后层
        self.lag_layer = MotorLagLayer(lag_mode=lag_mode, alpha_init=alpha_init)

        # 3) 冻结参数：物理主干参与前向，但不参与训练更新
        for param in self.phys.parameters():
            param.requires_grad_(False)
        self.gru_hidden_dim = hidden_dim
        self.alpha_dim = int(alpha_dim)
        self.use_u_init = bool(use_u_init)
        self.u_init_scale = float(u_init_scale)
        self.use_hist_init = bool(use_hist_init)
        self.hist_init_scale = float(hist_init_scale)
        self.use_actbank_alpha = bool(use_actbank_alpha)
        self.actbank_use_history = bool(actbank_use_history)
        self.actbank_alpha_scale = float(actbank_alpha_scale)
        self.actbank_taus_ms = tuple(float(tau_ms) for tau_ms in actbank_taus_ms)
        self.requires_history = bool(self.use_hist_init or self.actbank_use_history)
        if self.alpha_dim not in (1, 4):
            raise ValueError("alpha_dim must be 1 or 4")
        if self.use_actbank_alpha and self.alpha_dim != 1:
            raise ValueError("use_actbank_alpha requires alpha_dim == 1")
        if self.use_actbank_alpha and not self.actbank_taus_ms:
            raise ValueError("actbank_taus_ms must be non-empty when use_actbank_alpha=True")
        if self.use_actbank_alpha and any(tau_ms <= 0.0 for tau_ms in self.actbank_taus_ms):
            raise ValueError("actbank_taus_ms must be positive")

        # 4) 维度与结构检查
        state_dim = residual.out.out_features
        #state_dim：与状态/残差同维，12（最后一层Linear输出 12）。
        #  可学习模块：GRU 初始化、动态 alpha 头、GRUCell
        self.h_init = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())

        control_dim = 4
        feature_dim = state_dim + 3 * control_dim + hidden_dim
        self._validate_residual_input(
            state_dim=state_dim,
            control_feat_dim=3 * control_dim + hidden_dim,
        )
        #残差网络吃的是 [x_norm, u_raw, u_eff, u_raw-u_eff, h]——和无 GRU 版不同，无 GRU 时没有 h，只有 state_dim + 12。

        # 5) buffer：保存归一化常数，不参与优化器更新
        x_mean, x_scale = self._scaler_to_tensors(x_scaler, state_dim)
        u_mean, u_scale = self._scaler_to_tensors(u_scaler, control_dim)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("u_mean", u_mean)
        self.register_buffer("u_scale", u_scale)

        if self.use_u_init:
            self.u_init_head = nn.Sequential(
                nn.Linear(state_dim + control_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, control_dim),
            )
            nn.init.zeros_(self.u_init_head[-1].weight)
            nn.init.zeros_(self.u_init_head[-1].bias)
        else:
            self.u_init_head = None

        if self.use_hist_init:
            self.hist_encoder = nn.GRU(
                input_size=state_dim + control_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.hist_h_head = nn.Sequential(
                nn.Linear(hidden_dim + state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.hist_h_head[-1].weight)
            nn.init.zeros_(self.hist_h_head[-1].bias)
        else:
            self.hist_encoder = None
            self.hist_h_head = None

        if self.use_actbank_alpha:
            tau_sec = torch.tensor(self.actbank_taus_ms, dtype=torch.float32) / 1000.0
            lambdas = torch.exp(-torch.tensor(float(self.dt), dtype=torch.float32) / tau_sec)
            self.register_buffer("actbank_lambdas", lambdas.view(1, 1, -1))
            actbank_feat_dim = control_dim * len(self.actbank_taus_ms) + 3 * control_dim
            self.actbank_alpha_head = nn.Sequential(
                nn.Linear(actbank_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.actbank_alpha_head[-1].weight)
            nn.init.zeros_(self.actbank_alpha_head[-1].bias)
        else:
            self.actbank_lambdas = None
            self.actbank_alpha_head = None

        
        #h，形状 (B, hidden_dim)
        #由当前状态、指令、滞后 seed、GRU 记忆共同决定这一步的混合系数。
        self.alpha_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.alpha_dim),
        )
        nn.init.zeros_(self.alpha_head[-1].weight)#训练刚开始时，alpha_head 基本会输出接近 alpha_init 的常数。
        nn.init.constant_(self.alpha_head[-1].bias, math.log(alpha_init / (1.0 - alpha_init)))
        self.gru_cell = nn.GRUCell(feature_dim, hidden_dim)#h是可变的、有记忆的中间向量：每一步都会根据当前输入gru_in和上一步的h更新一次。


    #打包特征给其他地方用
    @staticmethod
    def _pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h):
        #输入：x_norm, u_raw_norm, u_eff_norm, h
        #输出：[x_norm, u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm, h]
        #用于将当前状态、指令、滞后 seed、GRU 记忆共同决定这一步的混合系数。
        return torch.cat(
            [x_norm, u_raw_norm, u_eff_norm, u_raw_norm - u_eff_norm, h],
            dim=-1,
        )

    def _actbank_update(self, bank_state, u_norm):
        if self.actbank_lambdas is None:
            raise ValueError("actbank_lambdas are unavailable when use_actbank_alpha=False")
        u = u_norm.unsqueeze(-1)
        return self.actbank_lambdas * bank_state + (1.0 - self.actbank_lambdas) * u

    def _actbank_init(self, u0_norm, u_hist=None):
        if self.actbank_lambdas is None:
            raise ValueError("actbank_lambdas are unavailable when use_actbank_alpha=False")

        if self.actbank_use_history:
            if u_hist is None:
                raise ValueError("u_hist is required when actbank_use_history=True")
            if u_hist.ndim != 3:
                raise ValueError("u_hist must have shape (B,L,4)")
            if u_hist.shape[0] != u0_norm.shape[0]:
                raise ValueError("u_hist batch size must match x0 batch size")
            if u_hist.shape[-1] != u0_norm.shape[-1]:
                raise ValueError("u_hist last dimension must be 4")
            if u_hist.shape[1] <= 0:
                raise ValueError("u_hist must contain at least one history step")

            bank = u_hist[:, 0, :].unsqueeze(-1).expand(
                -1, -1, self.actbank_lambdas.shape[-1]
            ).contiguous()
            for step in range(u_hist.shape[1]):
                bank = self._actbank_update(bank, u_hist[:, step, :])
            return bank

        return u0_norm.unsqueeze(-1).expand(
            -1, -1, self.actbank_lambdas.shape[-1]
        ).contiguous()




    def forward(self, x0, u_seq, x_hist=None, u_hist=None):
        # 兼容单步控制输入 (B,4)；统一成时序输入 (B,T,4) 做 rollout。
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        # 兼容 (B,1,12) 与 (B,12) 两种初始状态写法。
        if x0.ndim == 3:
            x_norm = x0.squeeze(1)
        else:
            x_norm = x0

        _, horizon, _ = u_seq.shape
        preds = []
        # 从当前状态生成初始记忆h；执行器内部状态用第一步控制作seed。
        h = self.h_init(x_norm)

        if self.use_hist_init:
            if x_hist is None or u_hist is None:
                raise ValueError("x_hist and u_hist are required when use_hist_init=True")
            if x_hist.ndim != 3 or u_hist.ndim != 3:
                raise ValueError("x_hist and u_hist must be 3D tensors")
            if x_hist.shape[0] != x_norm.shape[0] or u_hist.shape[0] != x_norm.shape[0]:
                raise ValueError("history batch size mismatch")
            if x_hist.shape[-1] != x_norm.shape[-1]:
                raise ValueError("x_hist last dimension must match state dim")
            if u_hist.shape[-1] != u_seq.shape[-1]:
                raise ValueError("u_hist last dimension must match control dim")
            if x_hist.shape[1] != u_hist.shape[1] + 1:
                raise ValueError("x_hist must have length L+1 and u_hist length L")
            if not torch.allclose(x_hist[:, -1, :], x_norm, atol=1e-6, rtol=1e-6):
                raise ValueError("x_hist must end at x0")

            hist_input = torch.cat([x_hist[:, :-1, :], u_hist], dim=-1)
            _, hist_hidden = self.hist_encoder(hist_input)
            e_hist = hist_hidden[-1]
            # Optional history-conditioned initialization of the observer hidden state.
            # The zero-initialized head makes this path initially identical to h0 = h_init(x0).
            delta_h = self.hist_init_scale * torch.tanh(
                self.hist_h_head(torch.cat([e_hist, x_norm], dim=-1))
            )
            h = h + delta_h

        u0_norm = u_seq[:, 0, :]

        # Optional learned initialization of the effective actuator state.
        # Zero-initialized head makes this path initially identical to u_eff0 = u0.
        if self.use_u_init:
            init_feat = torch.cat([x_norm, u0_norm], dim=-1)
            delta_u0_norm = self.u_init_scale * torch.tanh(self.u_init_head(init_feat))
            u_eff0_norm = u0_norm + delta_u0_norm
            u_eff_prev_real = self.u_denorm(u_eff0_norm)
        else:
            u_eff_prev_real = self.u_denorm(u0_norm)

        if self.use_actbank_alpha:
            if self.actbank_use_history and u_hist is None:
                raise ValueError(
                    "u_hist is required when use_actbank_alpha and actbank_use_history"
                )
            bank_state = self._actbank_init(u0_norm=u0_norm, u_hist=u_hist)
        else:
            bank_state = None

        for t in range(horizon):
            '''1取当前raw电机指令'''
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)

            # 先用固定一阶lag得到一个seed（先用老办法算一个u_eff_seed），再由GRU上下文预测动态alpha_t。
            u_eff_prev_norm = self.u_normed(u_eff_prev_real)
            u_eff_seed_real = self.lag_layer(u_eff_prev_real, u_raw_real)
            u_eff_seed_norm = self.u_normed(u_eff_seed_real)
            alpha_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_seed_norm, h)
            # alpha_dim=1: shared dynamic alpha over motors
            # alpha_dim=4: motor-wise dynamic alpha
            alpha_logits = self.alpha_head(alpha_in)

            if self.use_actbank_alpha:
                bank_flat = bank_state.reshape(bank_state.shape[0], -1)
                bank_feat = torch.cat(
                    [
                        bank_flat,
                        u_raw_norm,
                        u_eff_prev_norm,
                        u_raw_norm - u_eff_prev_norm,
                    ],
                    dim=-1,
                )
                # Optional actuator-memory correction for alpha logits only.
                # The zero-initialized head keeps this path initially identical to lag_gru.
                delta_alpha_logits = self.actbank_alpha_scale * torch.tanh(
                    self.actbank_alpha_head(bank_feat)
                )
                alpha_logits = alpha_logits + delta_alpha_logits

            alpha_t = torch.sigmoid(alpha_logits)

            # 最终有效电机转速：在上一时刻执行器状态和当前 raw 指令之间动态插值。
            u_eff_real = alpha_t * u_eff_prev_real + (1.0 - alpha_t) * u_raw_real
            u_eff_norm = self.u_normed(u_eff_real)

            # 物理主干在真实量纲下推进一步，再映回归一化空间。
            x_real = self.x_denorm(x_norm)
            x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            # 记忆状态 h 随当前状态、控制和有效执行器输入一起更新。
            gru_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            h = self.gru_cell(gru_in, h)

            # 残差网络学习 physics 未覆盖的修正量，再和物理预测相加。
            residual_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            # 若残差分支数值炸掉，则退回纯 physics 预测，保证 rollout 可继续。
            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            # 自回归 rollout：本步输出作为下一步输入继续往前滚。
            preds.append(x_next_norm.unsqueeze(1))
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real
            if self.use_actbank_alpha:
                bank_state = self._actbank_update(bank_state, u_raw_norm)

        # 拼成完整预测序列 (B,T,12)。
        return torch.cat(preds, dim=1)


class LagPhysResGRUControlModel(LagPhysResGRUModel):
    """Lag-GRU with a lightweight future-control context encoder."""

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        hidden_dim=64,
        control_ctx_dim=32,
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
        self.control_ctx_dim = int(control_ctx_dim)
        self.control_encoder = ControlTCNEncoder(ctx_dim=self.control_ctx_dim)
        self.ctx_h0_adapter = nn.Linear(self.control_ctx_dim, hidden_dim)
        self.ctx_step_adapter = nn.Linear(self.control_ctx_dim, hidden_dim)
        nn.init.zeros_(self.ctx_h0_adapter.weight)
        nn.init.zeros_(self.ctx_h0_adapter.bias)
        nn.init.zeros_(self.ctx_step_adapter.weight)
        nn.init.zeros_(self.ctx_step_adapter.bias)

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
        c_seq, c_global = self.control_encoder(u_seq)
        h = self.h_init(x_norm)
        h = h + self.ctx_h0_adapter(c_global)
        u_eff_prev_real = self.u_denorm(u_seq[:, 0, :])

        for t in range(horizon):
            u_raw_norm = u_seq[:, t, :]
            u_raw_real = self.u_denorm(u_raw_norm)
            c_t = c_seq[:, t, :]

            u_eff_seed_real = self.lag_layer(u_eff_prev_real, u_raw_real)
            u_eff_seed_norm = self.u_normed(u_eff_seed_real)
            h_alpha = h + self.ctx_step_adapter(c_t)
            alpha_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_seed_norm, h_alpha)
            alpha_t = torch.sigmoid(self.alpha_head(alpha_in))

            u_eff_real = alpha_t * u_eff_prev_real + (1.0 - alpha_t) * u_raw_real
            u_eff_norm = self.u_normed(u_eff_real)

            x_real = self.x_denorm(x_norm)
            x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)
            x_phys_next_norm = self.x_normed(x_phys_next_real)

            gru_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            h = self.gru_cell(gru_in, h)

            h_res = h + self.ctx_step_adapter(c_t)
            residual_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h_res)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_phys_next_norm + dx_res

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_phys_next_norm)

            preds.append(x_next_norm.unsqueeze(1))
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        return torch.cat(preds, dim=1)


class LagPhysResGRUTorqueModel(LagPhysResGRUModel):
    """GRU-conditioned lag model with learned residual body torque."""

    def __init__(
        self,
        phys,
        residual,
        x_scaler,
        u_scaler,
        lag_mode="per_motor",
        alpha_init=0.85,
        hidden_dim=64,
        torque_scale_factor: float = 0.2,
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
        self.torque_scale_factor = float(torque_scale_factor)

        self.torque_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.torque_head[-1].weight)
        nn.init.zeros_(self.torque_head[-1].bias)
        self.register_buffer(
            "torque_scale",
            self.torque_scale_factor * self.phys.max_torque.detach().clone().view(1, 3),
        )

    def forward(self, x0, u_seq, return_torque: bool = False):
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
        torque_preds = [] if return_torque else None
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
            x_phys_next_real = self.physics_step_from_motors(x_real, u_eff_real)

            gru_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            h = self.gru_cell(gru_in, h)

            delta_tau_b = self.torque_scale * torch.tanh(self.torque_head(h))
            x_torque_next_real = self.phys.apply_torque(
                x_real,
                delta_tau_b,
                x_phys_next_real=x_phys_next_real,
            )
            x_torque_next_norm = self.x_normed(x_torque_next_real)

            residual_in = self._pack_gru_features(x_norm, u_raw_norm, u_eff_norm, h)
            dx_res = self.residual.out(self.residual.mlp(residual_in))
            x_next_norm = x_torque_next_norm + dx_res

            finite_mask = torch.isfinite(x_next_norm).all(dim=-1, keepdim=True)
            x_next_norm = torch.where(finite_mask, x_next_norm, x_torque_next_norm)

            preds.append(x_next_norm.unsqueeze(1))
            if torque_preds is not None:
                torque_preds.append(delta_tau_b.unsqueeze(1))
            x_norm = x_next_norm
            u_eff_prev_real = u_eff_real

        pred_seq = torch.cat(preds, dim=1)
        if not return_torque:
            return pred_seq

        torque_pred_seq = torch.cat(torque_preds, dim=1)
        return pred_seq, torque_pred_seq


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
        )#输入维 12 + 4 = 16，输入维 4。

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        #输入维 hidden_dim：在 forward 里接的是 GRU 隐状态 h（与父类里的 gru_hidden_dim 一致，默认 64）。
        #输出 3 维：机体坐标系下的残差外力Δf_b，交给物理里的 apply_force
        #在电机动力学之外再补一项力。
       
       #「新分支从零扰动开始」，避免随机初始化破坏物理 rollout。
        nn.init.zeros_(self.u_init_head[-1].weight)
        nn.init.zeros_(self.u_init_head[-1].bias)
        nn.init.zeros_(self.force_head[-1].weight)
        nn.init.zeros_(self.force_head[-1].bias)

    def forward(self, x0, u_seq, return_force: bool = False):
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
    "ControlTCNEncoder",
    "LagPhysResQuadModel",
    "LagPhysResGRUModel",
    "LagPhysResGRUControlModel",
    "LagPhysResGRUTorqueModel",
    "LagPhysResGRUForceModel",
]
