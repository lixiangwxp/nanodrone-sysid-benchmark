import os

import joblib
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from dataset.dataset import quat_to_so3_log


class QuadDatasetWithAux(Dataset):
    def __init__(
        self,
        df,
        horizon=1,
        aux_cols=None,
        use_acc_aux=True,
        aux_candidate_cols=None,
    ):
        """
        Dataset with the same 12D state target as the baseline, plus aux supervision.

        Returns:
            x0: (1, 12)
            u_seq: (H, 4)
            x_seq: (H, 12)
            aux_seq: (H, aux_dim)
        """

        # --- Build state (12D) ---
        pos = df[["x", "y", "z"]].values
        vel = df[["vx", "vy", "vz"]].values
        omega = df[["wx", "wy", "wz"]].values
        quat = df[["qx", "qy", "qz", "qw"]].values
        rot_repr = quat_to_so3_log(torch.from_numpy(quat).float()).numpy()
        state = np.hstack([pos, vel, rot_repr, omega])
        num_steps = len(state)

        # --- Inputs ---
        u = df[["m1_rads", "m2_rads", "m3_rads", "m4_rads"]].values

        # --- Auxiliary supervision ---
        aux_signal, resolved_aux_cols = self._build_aux_signal(
            df=df,
            aux_cols=aux_cols,
            use_acc_aux=use_acc_aux,
            aux_candidate_cols=aux_candidate_cols,
        )

        if horizon == "full":
            horizon = num_steps - 1
        else:
            horizon = int(horizon)

        xs, us_seq, xs_seq, aux_seq = [], [], [], []
        for i in range(num_steps - horizon):
            xs.append(state[i].reshape(1, -1))
            us_seq.append(u[i : i + horizon])
            xs_seq.append(state[i + 1 : i + 1 + horizon])
            aux_seq.append(aux_signal[i + 1 : i + 1 + horizon])

        self.xs = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.us_seq = torch.tensor(np.stack(us_seq), dtype=torch.float32)
        self.xs_seq = torch.tensor(np.stack(xs_seq), dtype=torch.float32)
        self.aux_seq = torch.tensor(np.stack(aux_seq), dtype=torch.float32)
        self.aux_cols = resolved_aux_cols

    @staticmethod
    def _build_aux_signal(df, aux_cols=None, use_acc_aux=True, aux_candidate_cols=None):
        if aux_candidate_cols is None:
            aux_candidate_cols = ["az_body", "a_z_body", "acc_z_body", "a_body_z"]

        auto_detect = aux_cols is None
        if aux_cols is None:
            aux_cols = ["az_body"]
        elif isinstance(aux_cols, str):
            aux_cols = [aux_cols]

        missing = [col for col in aux_cols if col not in df.columns]
        if missing:
            if auto_detect and use_acc_aux:
                for candidate in aux_candidate_cols:
                    if candidate in df.columns:
                        aux_cols = [candidate]
                        missing = []
                        break

            if missing:
                if auto_detect and use_acc_aux:
                    raise ValueError(
                        "Could not find any auxiliary acceleration column. "
                        f"Tried default {['az_body']} and fallback candidates "
                        f"{aux_candidate_cols}. Available columns: {df.columns.tolist()}"
                    )
                raise ValueError(
                    "Missing auxiliary columns "
                    f"{missing}. Available columns: {df.columns.tolist()}"
                )

        return df[aux_cols].values, aux_cols

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.us_seq[idx], self.xs_seq[idx], self.aux_seq[idx]


def combine_concat_dataset_with_aux(
    concat_dataset,
    scale=False,
    fold="train",
    scaler_dir="./scalers",
    scale_aux=True,
):
    """
    Combine multiple aux-aware datasets into one dataset with optional scaling.
    """
    if not isinstance(concat_dataset, ConcatDataset):
        raise TypeError("Input must be a ConcatDataset.")
    assert fold in ["train", "valid", "test"]

    os.makedirs(scaler_dir, exist_ok=True)

    all_xs, all_us_seq, all_xs_seq, all_aux_seq = [], [], [], []
    for ds in concat_dataset.datasets:
        xs, us_seq, xs_seq, aux_seq = [], [], [], []
        for i in range(len(ds)):
            x, u, xseq, aux = ds[i]
            xs.append(x)
            us_seq.append(u)
            xs_seq.append(xseq)
            aux_seq.append(aux)

        all_xs.append(torch.stack(xs))
        all_us_seq.append(torch.stack(us_seq))
        all_xs_seq.append(torch.stack(xs_seq))
        all_aux_seq.append(torch.stack(aux_seq))

    final_xs = torch.cat(all_xs, dim=0)
    final_us_seq = torch.cat(all_us_seq, dim=0)
    final_xs_seq = torch.cat(all_xs_seq, dim=0)
    final_aux_seq = torch.cat(all_aux_seq, dim=0)

    print("✅ Combined dataset shapes:")
    print(f"  x0:      {final_xs.shape}")
    print(f"  u_seq:   {final_us_seq.shape}")
    print(f"  x_seq:   {final_xs_seq.shape}")
    print(f"  aux_seq: {final_aux_seq.shape}")

    if scale:
        x_scaler_path = os.path.join(scaler_dir, "x_scaler.pkl")
        u_scaler_path = os.path.join(scaler_dir, "u_scaler.pkl")
        aux_scaler_path = os.path.join(scaler_dir, "aux_scaler.pkl")

        if fold == "train":
            from sklearn.preprocessing import StandardScaler

            x_scaler = StandardScaler()
            u_scaler = StandardScaler()
            aux_scaler = StandardScaler() if scale_aux else None

            x_flat = np.concatenate(
                [
                    final_xs.reshape(-1, final_xs.shape[-1]).numpy(),
                    final_xs_seq.reshape(-1, final_xs_seq.shape[-1]).numpy(),
                ],
                axis=0,
            )
            u_flat = final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy()
            aux_flat = final_aux_seq.reshape(-1, final_aux_seq.shape[-1]).numpy()

            x_scaler.fit(x_flat)
            u_scaler.fit(u_flat)
            if scale_aux:
                aux_scaler.fit(aux_flat)

            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(u_scaler, u_scaler_path)
            if scale_aux:
                joblib.dump(aux_scaler, aux_scaler_path)
        else:
            x_scaler = joblib.load(x_scaler_path)
            u_scaler = joblib.load(u_scaler_path)
            aux_scaler = joblib.load(aux_scaler_path) if scale_aux else None

        final_xs = torch.from_numpy(
            x_scaler.transform(final_xs.reshape(-1, final_xs.shape[-1]).numpy())
        ).float()
        final_us_seq = torch.from_numpy(
            u_scaler.transform(final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy())
        ).float().reshape_as(final_us_seq)
        final_xs_seq = torch.from_numpy(
            x_scaler.transform(final_xs_seq.reshape(-1, final_xs_seq.shape[-1]).numpy())
        ).float().reshape_as(final_xs_seq)
        if scale_aux:
            final_aux_seq = torch.from_numpy(
                aux_scaler.transform(final_aux_seq.reshape(-1, final_aux_seq.shape[-1]).numpy())
            ).float().reshape_as(final_aux_seq)
    else:
        x_scaler = None
        u_scaler = None
        aux_scaler = None

    class CombinedDatasetWithAux(torch.utils.data.Dataset):
        def __init__(self, xs, us_seq, xs_seq, aux_seq, x_scaler=None, u_scaler=None, aux_scaler=None):
            self.xs = xs
            self.us_seq = us_seq
            self.xs_seq = xs_seq
            self.aux_seq = aux_seq
            self.x_scaler = x_scaler
            self.u_scaler = u_scaler
            self.aux_scaler = aux_scaler

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, idx):
            return self.xs[idx], self.us_seq[idx], self.xs_seq[idx], self.aux_seq[idx]

    return CombinedDatasetWithAux(
        final_xs,
        final_us_seq,
        final_xs_seq,
        final_aux_seq,
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        aux_scaler=aux_scaler,
    )


__all__ = ["QuadDatasetWithAux", "combine_concat_dataset_with_aux"]
