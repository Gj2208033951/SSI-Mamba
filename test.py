#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# =========================================================
# 导入训练模块
# 假设 model_train_code.py 在同目录下
# =========================================================
from Mamba_after import MultiModalDataset, MultiModalSiameseModel, ModelConfig

# =========================================================
# 配置
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 测试集 npz 文件路径（根据你的实际路径修改）
test_seq_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/seq_npz/test_seq.npz"
test_bw_npz = "/public/home/sy_gj_202807/silence/CNN_Mamba/data/mESC/train_data_multimodal_mESC/bw_npz/test_bw.npz"

# 训练好的模型权重
model_weight_path = "/public/home/sy_gj_202807/silence/CNN_Mamba/train/train_K562_multimodal_concat_mlp/best_model.pth"

# 输出目录
save_dir = "/public/home/sy_gj_202807/silence/CNN_Mamba/test/zero_K562_to_mESC"
os.makedirs(save_dir, exist_ok=True)
test_metrics_path = os.path.join(save_dir, "test_metrics.xlsx")
test_result_path = os.path.join(save_dir, "test_result.xlsx")
test_wrong_samples_path = os.path.join(save_dir, "test_wrong_samples.xlsx")

# =========================================================
# Dataset & DataLoader
# =========================================================
test_dataset = MultiModalDataset(test_seq_npz, test_bw_npz)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# =========================================================
# Model
# =========================================================
config = ModelConfig()
model = MultiModalSiameseModel(config).to(device)
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# =========================================================
# 测试推理
# =========================================================
all_labels = []
all_preds_prob = []
all_fromid = []

with torch.no_grad():
    for x1, x2, bw1, bw2, bw1_valid, bw2_valid, y, fromid in test_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        bw1 = bw1.to(device)
        bw2 = bw2.to(device)
        bw1_valid = bw1_valid.to(device)
        bw2_valid = bw2_valid.to(device)
        y = y.to(device)

        logits = model(x1, x2, bw1, bw2, bw1_valid, bw2_valid)
        preds_prob = torch.sigmoid(logits)

        all_labels.append(y.cpu().numpy())
        all_preds_prob.append(preds_prob.cpu().numpy())
        all_fromid.append(np.array(fromid))

# 合并所有批次
y_true = np.concatenate(all_labels)
y_pred_prob = np.concatenate(all_preds_prob)
y_pred = (y_pred_prob > 0.5).astype(float)
fromid = np.concatenate(all_fromid)

# =========================================================
# 计算指标
# =========================================================
test_acc = accuracy_score(y_true, y_pred)
test_prec = precision_score(y_true, y_pred, zero_division=0)
test_rec = recall_score(y_true, y_pred, zero_division=0)
test_f1 = f1_score(y_true, y_pred, zero_division=0)
test_auroc = roc_auc_score(y_true, y_pred_prob)
test_auprc = average_precision_score(y_true, y_pred_prob)

# 保存测试指标
df_metrics = pd.DataFrame([{
    "Accuracy": test_acc,
    "Precision": test_prec,
    "Recall": test_rec,
    "F1": test_f1,
    "AUROC": test_auroc,
    "AUPRC": test_auprc
}])
df_metrics.to_excel(test_metrics_path, index=False)

# 保存测试预测结果
df_result = pd.DataFrame({
    "FromID": fromid,
    "y_true": y_true,
    "y_pred": y_pred,
    "y_prob": y_pred_prob
})
df_result.to_excel(test_result_path, index=False)

# 保存错误预测样本
wrong_indices = np.where(y_pred != y_true)[0]
wrong_fromid = fromid[wrong_indices]
df_wrong = pd.DataFrame({"FromID": wrong_fromid})
df_wrong.to_excel(test_wrong_samples_path, index=False)

print("Test finished.")
print(f"Accuracy: {test_acc:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
print(f"Metrics saved to: {test_metrics_path}")
print(f"Results saved to: {test_result_path}")
print(f"Wrong samples saved to: {test_wrong_samples_path}")