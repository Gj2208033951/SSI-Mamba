#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from mamba_ssm.modules.mamba_simple import Mamba

warnings.filterwarnings("ignore")


# =========================================================
# 0. 固定随机种子
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# =========================================================
# 1. 配置
# =========================================================
class ModelConfig:
    # -------- 数据长度 --------
    seq_len: int = 200
    bw_len: int = 200

    # -------- 输入通道 --------
    dna_in_channels: int = 6      # A/C/G/T/N + anchor mask
    bw_in_channels: int = 6       # 6个组蛋白通道

    # -------- DNA分支（Mamba）--------
    conv_k1: int = 5
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 4

    # -------- bigWig分支（Transformer）--------
    bw_d_model: int = 128
    bw_nhead: int = 8
    bw_num_layers: int = 2
    bw_ffn_dim: int = 256

    # -------- 融合 / 分类 --------
    fusion_hidden: int = 256
    dropout: float = 0.2


# =========================================================
# 2. 基础模块
# =========================================================
class MambaBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class CrossStateInteraction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model

        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)

        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_a, h_b: [B, L, d_model]
        """
        h_a_pooled = h_a.mean(dim=1, keepdim=True)   # [B, 1, d_model]
        h_b_pooled = h_b.mean(dim=1, keepdim=True)

        # a query b
        q_a = self.query_proj(h_a_pooled)
        k_b = self.key_proj(h_b)
        v_b = self.value_proj(h_b)

        attn_weights_ab = torch.matmul(q_a, k_b.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_weights_ab = F.softmax(attn_weights_ab, dim=-1)
        context_b = torch.matmul(attn_weights_ab, v_b)   # [B, 1, d_model]

        # b query a
        q_b = self.query_proj(h_b_pooled)
        k_a = self.key_proj(h_a)
        v_a = self.value_proj(h_a)

        attn_weights_ba = torch.matmul(q_b, k_a.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_weights_ba = F.softmax(attn_weights_ba, dim=-1)
        context_a = torch.matmul(attn_weights_ba, v_a)   # [B, 1, d_model]

        gate_a = self.gate(torch.cat([h_a_pooled, context_b], dim=-1))
        gate_b = self.gate(torch.cat([h_b_pooled, context_a], dim=-1))

        h_a_new = h_a + self.dropout(gate_a * context_b.expand_as(h_a))
        h_b_new = h_b + self.dropout(gate_b * context_a.expand_as(h_b))

        return self.norm(h_a_new), self.norm(h_b_new)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =========================================================
# 3. DNA分支：仅序列 -> CNN -> Mamba
# =========================================================
class DNAMambaTower(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.dna_input_layer = nn.Sequential(
            ConvBlock(
                in_channels=config.dna_in_channels,
                out_channels=config.d_model,
                kernel_size=config.conv_k1,
                dropout=config.dropout
            )
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.seq_len, config.d_model) * 0.02
        )

        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, dna_x: torch.Tensor) -> torch.Tensor:
        """
        dna_x: [B, C_seq, L_seq] = [B, 6, L]
        return: [B, L, d_model]
        """
        dna_x = dna_x.float()                               # [B, 6, L]
        dna_feat = self.dna_input_layer(dna_x)             # [B, d_model, L]
        x = dna_feat.transpose(1, 2)                       # [B, L, d_model]

        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class DNABranch(nn.Module):
    """
    双端DNA分支：
    seq_a, seq_b -> 共享 DNAMambaTower -> CrossStateInteraction -> pooled pair feature
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tower = DNAMambaTower(config)

        self.cross_interactions = nn.ModuleList([
            CrossStateInteraction(config) for _ in range(2)
        ])

        self.pooling = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.out_dim = config.d_model * 4

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        h_a = self.tower(seq_a)   # [B, L, d_model]
        h_b = self.tower(seq_b)

        for cross_layer in self.cross_interactions:
            h_a, h_b = cross_layer(h_a, h_b)

        p_a_mean = h_a.mean(dim=1)
        p_a_max = h_a.max(dim=1).values

        p_b_mean = h_b.mean(dim=1)
        p_b_max = h_b.max(dim=1).values

        p_a = self.pooling(p_a_mean + p_a_max)   # [B, d_model]
        p_b = self.pooling(p_b_mean + p_b_max)

        combined = torch.cat(
            [p_a, p_b, p_a * p_b, torch.abs(p_a - p_b)],
            dim=-1
        )  # [B, 4*d_model]

        return combined


# =========================================================
# 4. bigWig分支：双端6通道 -> Transformer Encoder
# =========================================================
class BigWigTransformerTower(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.bw_in_channels, config.bw_d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.bw_len, config.bw_d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.bw_d_model,
            nhead=config.bw_nhead,
            dim_feedforward=config.bw_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.bw_num_layers
        )

        self.norm = nn.LayerNorm(config.bw_d_model)

    def forward(self, bw_x: torch.Tensor, bw_valid: torch.Tensor) -> torch.Tensor:
        """
        bw_x:     [B, 6, L_bw]
        bw_valid: [B, L_bw]
        return:   [B, L_bw, bw_d_model]
        """
        # [B, 6, L] -> [B, L, 6]
        x = bw_x.float().transpose(1, 2)

        # [B, L, 6] -> [B, L, bw_d_model]
        x = self.input_proj(x)

        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # src_key_padding_mask: True 表示要mask的位置
        key_padding_mask = (bw_valid <= 0)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class BigWigBranch(nn.Module):
    """
    双端bigWig分支：
    BW_a, BW_b -> 共享 TransformerTower -> pooled pair feature
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tower = BigWigTransformerTower(config)

        self.pooling = nn.Sequential(
            nn.Linear(config.bw_d_model, config.bw_d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.out_dim = config.bw_d_model * 4

    @staticmethod
    def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        valid_mask: [B, L] with 1/0
        """
        mask = valid_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    @staticmethod
    def masked_max(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        valid_mask: [B, L]
        """
        mask = valid_mask.unsqueeze(-1).bool()
        x_masked = x.masked_fill(~mask, float("-inf"))
        out = x_masked.max(dim=1).values
        out[out == float("-inf")] = 0.0
        return out

    def forward(
        self,
        bw_a: torch.Tensor,
        bw_b: torch.Tensor,
        bw_valid_a: torch.Tensor,
        bw_valid_b: torch.Tensor
    ) -> torch.Tensor:
        h_a = self.tower(bw_a, bw_valid_a)   # [B, L, bw_d_model]
        h_b = self.tower(bw_b, bw_valid_b)

        p_a_mean = self.masked_mean(h_a, bw_valid_a)
        p_a_max = self.masked_max(h_a, bw_valid_a)

        p_b_mean = self.masked_mean(h_b, bw_valid_b)
        p_b_max = self.masked_max(h_b, bw_valid_b)

        p_a = self.pooling(p_a_mean + p_a_max)
        p_b = self.pooling(p_b_mean + p_b_max)

        combined = torch.cat(
            [p_a, p_b, p_a * p_b, torch.abs(p_a - p_b)],
            dim=-1
        )  # [B, 4*bw_d_model]

        return combined


# =========================================================
# 5. 多模态融合：Concat + MLP
# =========================================================
class FusionClassifier(nn.Module):
    def __init__(self, dna_dim: int, bw_dim: int, fusion_hidden: int, dropout: float):
        super().__init__()
        in_dim = dna_dim + bw_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden // 2, 1)
        )

    def forward(self, z_dna: torch.Tensor, z_bw: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_dna, z_bw], dim=-1)
        logits = self.mlp(z).squeeze(-1)
        return logits


# =========================================================
# 6. 完整模型
# =========================================================
class MultiModalSiameseModel(nn.Module):
    """
    第一个版本：
    DNA分支：双端序列 -> CNN -> Mamba -> CrossStateInteraction
    bigWig分支：双端6通道 -> Transformer
    融合：Concat + MLP
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.dna_branch = DNABranch(config)
        self.bw_branch = BigWigBranch(config)

        self.classifier = FusionClassifier(
            dna_dim=self.dna_branch.out_dim,
            bw_dim=self.bw_branch.out_dim,
            fusion_hidden=config.fusion_hidden,
            dropout=config.dropout
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        bw_a: torch.Tensor,
        bw_b: torch.Tensor,
        bw_valid_a: torch.Tensor,
        bw_valid_b: torch.Tensor
    ) -> torch.Tensor:
        z_dna = self.dna_branch(seq_a, seq_b)
        z_bw = self.bw_branch(bw_a, bw_b, bw_valid_a, bw_valid_b)
        logits = self.classifier(z_dna, z_bw)
        return logits


# =========================================================
# 7. Dataset
# seq_npz 和 bw_npz 分开读取
# =========================================================
class MultiModalDataset(Dataset):
    def __init__(
        self,
        seq_npz_path: str,
        bw_npz_path: str,
        subset_ratio: float = 1.0,
        seed: int = 42
    ):
        seq_data = np.load(seq_npz_path, allow_pickle=True)
        bw_data = np.load(bw_npz_path, allow_pickle=True)

        self.X1 = seq_data["X1"].astype(np.float32)           # [N, 6, L_seq]
        self.X2 = seq_data["X2"].astype(np.float32)
        self.X1_valid = seq_data["X1_valid"].astype(np.float32)
        self.X2_valid = seq_data["X2_valid"].astype(np.float32)
        self.y_seq = seq_data["y"].astype(np.float32)
        self.FromID_seq = seq_data["FromID"]

        self.BW1 = bw_data["BW1"].astype(np.float32)          # [N, 6, L_bw]
        self.BW2 = bw_data["BW2"].astype(np.float32)
        self.BW1_valid = bw_data["BW1_valid"].astype(np.float32)  # [N, L_bw]
        self.BW2_valid = bw_data["BW2_valid"].astype(np.float32)
        self.y_bw = bw_data["y"].astype(np.float32)
        self.FromID_bw = bw_data["FromID"]

        # 对齐检查
        if len(self.y_seq) != len(self.y_bw):
            raise ValueError("seq_npz 和 bw_npz 的样本数不一致")

        if not np.array_equal(self.y_seq, self.y_bw):
            raise ValueError("seq_npz 和 bw_npz 的标签顺序不一致")

        if len(self.FromID_seq) == len(self.FromID_bw):
            seq_ids = np.array([str(x) for x in self.FromID_seq])
            bw_ids = np.array([str(x) for x in self.FromID_bw])
            if not np.array_equal(seq_ids, bw_ids):
                raise ValueError("seq_npz 和 bw_npz 的 FromID 顺序不一致")

        self.y = self.y_seq
        self.FromID = self.FromID_seq

        # 子采样
        if subset_ratio < 1.0:
            np.random.seed(seed)
            total_samples = len(self.y)
            subset_size = max(1, int(total_samples * subset_ratio))
            indices = np.random.choice(total_samples, size=subset_size, replace=False)

            self.X1 = self.X1[indices]
            self.X2 = self.X2[indices]
            self.X1_valid = self.X1_valid[indices]
            self.X2_valid = self.X2_valid[indices]

            self.BW1 = self.BW1[indices]
            self.BW2 = self.BW2[indices]
            self.BW1_valid = self.BW1_valid[indices]
            self.BW2_valid = self.BW2_valid[indices]

            self.y = self.y[indices]
            self.FromID = self.FromID[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = torch.tensor(self.X1[idx], dtype=torch.float32)            # [6, L_seq]
        x2 = torch.tensor(self.X2[idx], dtype=torch.float32)

        bw1 = torch.tensor(self.BW1[idx], dtype=torch.float32)          # [6, L_bw]
        bw2 = torch.tensor(self.BW2[idx], dtype=torch.float32)

        bw1_valid = torch.tensor(self.BW1_valid[idx], dtype=torch.float32)  # [L_bw]
        bw2_valid = torch.tensor(self.BW2_valid[idx], dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.float32)

        from_id = self.FromID[idx]
        if isinstance(from_id, np.generic):
            from_id = from_id.item()
        from_id = str(from_id)

        return x1, x2, bw1, bw2, bw1_valid, bw2_valid, y, from_id


# =========================================================
# 8. 路径配置
# 你需要改成自己的实际路径
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
config = ModelConfig()

# ---------- 训练集 ----------
train_seq_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/seq_npz/train_seq.npz"
train_bw_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/bw_npz/train_bw.npz"

val_seq_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/seq_npz/val_seq.npz"
val_bw_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/bw_npz/val_bw.npz"

test_seq_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/seq_npz/test_seq.npz"
test_bw_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/bw_npz/test_bw.npz"

train_dataset = MultiModalDataset(train_seq_npz, train_bw_npz)
val_dataset = MultiModalDataset(val_seq_npz, val_bw_npz)
test_dataset = MultiModalDataset(test_seq_npz, test_bw_npz)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

model = MultiModalSiameseModel(config).to(device)

print("train X1 shape:", train_dataset.X1.shape, train_dataset.X1.dtype)
print("train BW1 shape:", train_dataset.BW1.shape, train_dataset.BW1.dtype)
print("sample x1 shape:", train_dataset[0][0].shape)
print("sample bw1 shape:", train_dataset[0][2].shape)
print("sample bw1_valid shape:", train_dataset[0][4].shape)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
best_val_auroc = -float("inf")
patience = 5
patience_counter = 0
epoch_results = []

save_dir = "/public/home/sy_gj_202807/silence/CNN_Mamba/train/train_mESC_multimodal_concat_ml"
os.makedirs(save_dir, exist_ok=True)

best_model_path = os.path.join(save_dir, "best_model.pth")
epoch_result_path = os.path.join(save_dir, "epoch_result.xlsx")
test_metrics_path = os.path.join(save_dir, "test_metrics.xlsx")
test_result_path = os.path.join(save_dir, "test_result.xlsx")
test_wrong_samples_path = os.path.join(save_dir, "test_wrong_samples.xlsx")


# =========================================================
# 9. 训练 + 验证
# =========================================================
for epoch in range(num_epochs):
    # -------------------
    # 训练
    # -------------------
    model.train()
    all_train_labels = []
    all_train_preds = []

    for x1_batch, x2_batch, bw1_batch, bw2_batch, bw1_valid_batch, bw2_valid_batch, y_batch, _ in train_loader:
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        bw1_batch = bw1_batch.to(device)
        bw2_batch = bw2_batch.to(device)
        bw1_valid_batch = bw1_valid_batch.to(device)
        bw2_valid_batch = bw2_valid_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            x1_batch, x2_batch,
            bw1_batch, bw2_batch,
            bw1_valid_batch, bw2_valid_batch
        )

        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(logits)
        all_train_labels.append(y_batch.detach().cpu().numpy())
        all_train_preds.append(preds.detach().cpu().numpy())

    y_true_train = np.concatenate(all_train_labels)
    y_pred_train_prob = np.concatenate(all_train_preds)
    y_pred_train = (y_pred_train_prob > 0.5).astype(float)

    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_prec = precision_score(y_true_train, y_pred_train, zero_division=0)
    train_rec = recall_score(y_true_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)
    train_auroc = roc_auc_score(y_true_train, y_pred_train_prob)
    train_auprc = average_precision_score(y_true_train, y_pred_train_prob)

    # -------------------
    # 验证
    # -------------------
    model.eval()
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for x1_batch, x2_batch, bw1_batch, bw2_batch, bw1_valid_batch, bw2_valid_batch, y_batch, _ in val_loader:
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            bw1_batch = bw1_batch.to(device)
            bw2_batch = bw2_batch.to(device)
            bw1_valid_batch = bw1_valid_batch.to(device)
            bw2_valid_batch = bw2_valid_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(
                x1_batch, x2_batch,
                bw1_batch, bw2_batch,
                bw1_valid_batch, bw2_valid_batch
            )
            preds = torch.sigmoid(logits)

            all_val_labels.append(y_batch.cpu().numpy())
            all_val_preds.append(preds.cpu().numpy())

    y_true_val = np.concatenate(all_val_labels)
    y_pred_val_prob = np.concatenate(all_val_preds)
    y_pred_val = (y_pred_val_prob > 0.5).astype(float)

    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_prec = precision_score(y_true_val, y_pred_val, zero_division=0)
    val_rec = recall_score(y_true_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
    val_auroc = roc_auc_score(y_true_val, y_pred_val_prob)
    val_auprc = average_precision_score(y_true_val, y_pred_val_prob)

    epoch_results.append({
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "train_prec": train_prec,
        "train_rec": train_rec,
        "train_f1": train_f1,
        "train_auroc": train_auroc,
        "train_auprc": train_auprc,
        "val_acc": val_acc,
        "val_prec": val_prec,
        "val_rec": val_rec,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        "val_auprc": val_auprc
    })

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f} | "
        f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} | "
        f"Train Recall: {train_rec:.4f}, Val Recall: {val_rec:.4f}"
    )

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch + 1}, Val AUROC: {val_auroc:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement in Val AUROC for {patience_counter} epoch(s)")
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break


# =========================================================
# 10. 保存训练过程指标
# =========================================================
df_epoch = pd.DataFrame(epoch_results)
df_epoch.to_excel(epoch_result_path, index=False)
print(f"Epoch results saved to: {epoch_result_path}")


# =========================================================
# 11. 测试集评估
# =========================================================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_test_labels = []
all_test_preds_prob = []
all_test_fromid = []

with torch.no_grad():
    for x1_batch, x2_batch, bw1_batch, bw2_batch, bw1_valid_batch, bw2_valid_batch, y_batch, fromid_batch in test_loader:
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        bw1_batch = bw1_batch.to(device)
        bw2_batch = bw2_batch.to(device)
        bw1_valid_batch = bw1_valid_batch.to(device)
        bw2_valid_batch = bw2_valid_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(
            x1_batch, x2_batch,
            bw1_batch, bw2_batch,
            bw1_valid_batch, bw2_valid_batch
        )

        preds_prob = torch.sigmoid(logits)

        all_test_labels.append(y_batch.cpu().numpy())
        all_test_preds_prob.append(preds_prob.cpu().numpy())
        all_test_fromid.append(np.array(fromid_batch))

y_true_test = np.concatenate(all_test_labels)
y_pred_test_prob = np.concatenate(all_test_preds_prob)
y_pred_test = (y_pred_test_prob > 0.5).astype(float)
fromid_test = np.concatenate(all_test_fromid)

test_acc = accuracy_score(y_true_test, y_pred_test)
test_prec = precision_score(y_true_test, y_pred_test, zero_division=0)
test_rec = recall_score(y_true_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
test_auroc = roc_auc_score(y_true_test, y_pred_test_prob)
test_auprc = average_precision_score(y_true_test, y_pred_test_prob)

# 保存测试指标
df_test_metrics = pd.DataFrame([{
    "Accuracy": test_acc,
    "Precision": test_prec,
    "Recall": test_rec,
    "F1": test_f1,
    "AUROC": test_auroc,
    "AUPRC": test_auprc
}])
df_test_metrics.to_excel(test_metrics_path, index=False)

# 保存测试预测结果
df_test = pd.DataFrame({
    "FromID": fromid_test,
    "y_true": y_true_test,
    "y_pred": y_pred_test,
    "y_prob": y_pred_test_prob
})
df_test.to_excel(test_result_path, index=False)

# 保存错误预测样本
wrong_indices = np.where(y_pred_test != y_true_test)[0]
wrong_fromid = fromid_test[wrong_indices]
df_wrong = pd.DataFrame({"FromID": wrong_fromid})
df_wrong.to_excel(test_wrong_samples_path, index=False)

print(
    f"Test metrics saved. "
    f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, "
    f"Recall: {test_rec:.4f}, F1: {test_f1:.4f}, "
    f"AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}"
)
print(f"Test metrics file: {test_metrics_path}")
print(f"Test result file: {test_result_path}")
print(f"Wrong samples file: {test_wrong_samples_path}")