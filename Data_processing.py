#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import pyBigWig

# =========================================================
# 配置
# =========================================================
input_file = "GM12878_P_N.xlsx"
fasta_file = r"/public/home/sy_gj_202807/silence/CNN_Mamba/data/fa/hg38.fa"

# 6个组蛋白 bigWig 文件
bigwig_files = {
    "H3K4me3": "GM12878_H3K4me3.bigWig",
    "H3K9me3": "GM12878_H3K9me3.bigWig",
    "H3K36me3": "GM12878_H3K36me3.bigWig",
    "H3K4me1": "GM12878_H3K4me1.bigWig",
    "H3K9ac": "GM12878_H3K9ac.bigWig",
    "H3K27ac": "GM12878_H3K27ac.bigWig",
}

# 输出目录
save_dir = "./train_data_multimodal_GM12878"
seq_save_dir = os.path.join(save_dir, "seq_npz")
bw_save_dir = os.path.join(save_dir, "bw_npz")
os.makedirs(seq_save_dir, exist_ok=True)
os.makedirs(bw_save_dir, exist_ok=True)

# 标签映射
label_map = {"N": 0, "P": 1}

# -----------------------------
# 手动指定 chr1 划分
# -----------------------------
train_chr = [
    "chr1", "chr7", "chr6", "chr5", "chr22", "chr4", "chr3", "chr2",
    "chr16", "chr15", "chr14", "chr13", "chr12", "chr11", "chr10",
    "chr18", "chrX"
]
val_chr = [
    "chr19", "chr9", "chr20", "chr21"
]
test_chr = [
    "chr8", "chr17"
]

# -----------------------------
# 窗口长度
# 你可以自行修改
# -----------------------------
seq_len1 = 200
seq_len2 = 200

bw_len1 = 200
bw_len2 = 200

# -----------------------------
# 输出格式设置
# True: 输出 [N, C, L]，适合 PyTorch Conv1d
# False: 输出 [N, L, C]
# -----------------------------
SEQ_CHANNEL_FIRST = True
BW_CHANNEL_FIRST = True

# -----------------------------
# bigWig 信号是否做 log1p
# 推荐 True
# -----------------------------
APPLY_LOG1P_TO_BIGWIG = True


# =========================================================
# 工具函数
# =========================================================
def get_center_window(chrom_len, start, end, max_len):
    """
    以 anchor 中心定义固定窗口，返回：
    win_start, win_end, left_pad, right_pad, valid_start, valid_end
    """
    center = (int(start) + int(end)) // 2
    half = max_len // 2
    win_start = center - half
    win_end = win_start + max_len

    left_pad = max(0, -win_start)
    right_pad = max(0, win_end - chrom_len)

    valid_start = max(0, win_start)
    valid_end = min(chrom_len, win_end)

    return win_start, win_end, left_pad, right_pad, valid_start, valid_end


def encode_dna_with_anchor_mask_from_genome(fasta, chrom, start, end, max_len):
    """
    输出:
        arr: [L, 6]
            前5通道: A,C,G,T,N
            第6通道: anchor mask
        valid_mask: [L]
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    arr = np.zeros((max_len, 6), dtype=np.float32)
    valid_mask = np.zeros(max_len, dtype=np.float32)

    if pd.isna(chrom) or pd.isna(start) or pd.isna(end):
        arr[:, 4] = 1.0
        return arr, valid_mask

    chrom = str(chrom)
    start = int(start)
    end = int(end)

    if chrom not in fasta.keys():
        arr[:, 4] = 1.0
        return arr, valid_mask

    chrom_len = len(fasta[chrom])

    win_start, win_end, left_pad, right_pad, valid_start, valid_end = get_center_window(
        chrom_len, start, end, max_len
    )

    # 提取真实序列，越界补 N
    if valid_start >= valid_end:
        seq_fixed = "N" * max_len
    else:
        seq = fasta[chrom][valid_start:valid_end].seq.upper()
        seq_fixed = "N" * left_pad + seq + "N" * right_pad

        if len(seq_fixed) < max_len:
            seq_fixed += "N" * (max_len - len(seq_fixed))
        elif len(seq_fixed) > max_len:
            seq_fixed = seq_fixed[:max_len]

    # 写入 one-hot 前5通道
    for i, c in enumerate(seq_fixed):
        arr[i, mapping.get(c, 4)] = 1.0

    # 有效位置 mask
    if (max_len - right_pad) > left_pad:
        valid_mask[left_pad:max_len - right_pad] = 1.0

    # anchor mask（第6通道）
    anchor_rel_start = start - win_start
    anchor_rel_end = end - win_start
    anchor_rel_start = max(0, anchor_rel_start)
    anchor_rel_end = min(max_len, anchor_rel_end)

    if anchor_rel_start < anchor_rel_end:
        arr[anchor_rel_start:anchor_rel_end, 5] = 1.0

    return arr, valid_mask


def extract_one_bigwig_from_genome_window(bw, chrom, start, end, max_len, apply_log1p=True):
    """
    对单个 bigWig 文件提取固定窗口信号
    输出:
        values_fixed: [L]
        valid_mask: [L]
    """
    values_fixed = np.zeros(max_len, dtype=np.float32)
    valid_mask = np.zeros(max_len, dtype=np.float32)

    if pd.isna(chrom) or pd.isna(start) or pd.isna(end):
        return values_fixed, valid_mask

    chrom = str(chrom)
    start = int(start)
    end = int(end)

    if chrom not in bw.chroms():
        return values_fixed, valid_mask

    chrom_len = bw.chroms(chrom)

    win_start, win_end, left_pad, right_pad, valid_start, valid_end = get_center_window(
        chrom_len, start, end, max_len
    )

    if valid_start >= valid_end:
        return values_fixed, valid_mask

    values = bw.values(chrom, valid_start, valid_end, numpy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if apply_log1p:
        values = np.log1p(values)

    insert_start = left_pad
    insert_end = left_pad + len(values)

    values_fixed[insert_start:insert_end] = values
    valid_mask[insert_start:insert_end] = 1.0

    return values_fixed, valid_mask


def extract_multi_bigwig_from_genome_window(bw_dict, chrom, start, end, max_len, apply_log1p=True):
    """
    对6个 bigWig 同时提取
    输出:
        values_all: [6, L]
        valid_mask: [L]
    说明:
        这里默认所有 bigWig 的 valid_mask 一样，
        实际以第一个成功读取的 bigWig 的有效区域为准。
    """
    marks = list(bw_dict.keys())
    num_marks = len(marks)

    values_all = np.zeros((num_marks, max_len), dtype=np.float32)
    valid_mask_final = np.zeros(max_len, dtype=np.float32)

    for i, mark in enumerate(marks):
        bw = bw_dict[mark]
        values_i, valid_mask_i = extract_one_bigwig_from_genome_window(
            bw, chrom, start, end, max_len, apply_log1p=apply_log1p
        )
        values_all[i] = values_i

        # 使用并集或首个mask都可以，这里用并集更稳
        valid_mask_final = np.maximum(valid_mask_final, valid_mask_i)

    return values_all, valid_mask_final


def maybe_transpose_seq(x):
    """
    输入 x: [N, L, C]
    输出:
        若 SEQ_CHANNEL_FIRST=True -> [N, C, L]
        否则保持 [N, L, C]
    """
    if SEQ_CHANNEL_FIRST:
        return np.transpose(x, (0, 2, 1))
    return x


def maybe_transpose_bw(x):
    """
    输入 x: [N, C, L]
    输出:
        若 BW_CHANNEL_FIRST=True -> 保持 [N, C, L]
        否则 -> [N, L, C]
    """
    if BW_CHANNEL_FIRST:
        return x
    return np.transpose(x, (0, 2, 1))


# =========================================================
# 读取 Excel
# =========================================================
df = pd.read_excel(input_file)
df["Label_num"] = df["Label"].map(label_map)

required_columns = [
    "chr1", "start1", "end1",
    "chr2", "start2", "end2",
    "Label", "FromID"
]
missing_columns = [c for c in required_columns if c not in df.columns]
if len(missing_columns) > 0:
    raise ValueError(f"Excel 缺少以下必要列: {missing_columns}")

all_chr = set(df["chr1"].dropna().unique())
assigned_chr = set(train_chr + val_chr + test_chr)
missing_chr = all_chr - assigned_chr
if len(missing_chr) > 0:
    print("⚠️ 以下 chr1 未被分配:", missing_chr)


# =========================================================
# 打开 fasta 和 6个 bigWig
# =========================================================
print("正在打开参考基因组...")
fasta = Fasta(fasta_file)

print("正在打开 bigWig 文件...")
bw_dict = {}
for mark, bw_path in bigwig_files.items():
    if not os.path.exists(bw_path):
        raise FileNotFoundError(f"bigWig 文件不存在: {bw_path}")
    bw_dict[mark] = pyBigWig.open(bw_path)

print("已打开以下 bigWig:")
for k, v in bigwig_files.items():
    print(f"  {k}: {v}")


# =========================================================
# 处理子集
# =========================================================
def process_subset(chroms, name):
    subset = df[df["chr1"].isin(chroms)].reset_index(drop=True)

    if len(subset) == 0:
        print(f"⚠️ {name} 集合为空，跳过")
        return

    # -------------------------
    # anchor1
    # -------------------------
    x1_list = []
    x1_valid_list = []
    bw1_list = []
    bw1_valid_list = []

    for chrom, start, end in zip(subset["chr1"], subset["start1"], subset["end1"]):
        x1, x1_valid = encode_dna_with_anchor_mask_from_genome(
            fasta, chrom, start, end, seq_len1
        )
        bw1, bw1_valid = extract_multi_bigwig_from_genome_window(
            bw_dict, chrom, start, end, bw_len1,
            apply_log1p=APPLY_LOG1P_TO_BIGWIG
        )

        x1_list.append(x1)              # [L_seq, 6]
        x1_valid_list.append(x1_valid)  # [L_seq]
        bw1_list.append(bw1)            # [6, L_bw]
        bw1_valid_list.append(bw1_valid)  # [L_bw]

    X1 = np.stack(x1_list).astype(np.float32)               # [N, L_seq, 6]
    X1_valid = np.stack(x1_valid_list).astype(np.float32)   # [N, L_seq]
    BW1 = np.stack(bw1_list).astype(np.float32)             # [N, 6, L_bw]
    BW1_valid = np.stack(bw1_valid_list).astype(np.float32) # [N, L_bw]

    # -------------------------
    # anchor2
    # -------------------------
    x2_list = []
    x2_valid_list = []
    bw2_list = []
    bw2_valid_list = []

    for chrom, start, end in zip(subset["chr2"], subset["start2"], subset["end2"]):
        x2, x2_valid = encode_dna_with_anchor_mask_from_genome(
            fasta, chrom, start, end, seq_len2
        )
        bw2, bw2_valid = extract_multi_bigwig_from_genome_window(
            bw_dict, chrom, start, end, bw_len2,
            apply_log1p=APPLY_LOG1P_TO_BIGWIG
        )

        x2_list.append(x2)
        x2_valid_list.append(x2_valid)
        bw2_list.append(bw2)
        bw2_valid_list.append(bw2_valid)

    X2 = np.stack(x2_list).astype(np.float32)               # [N, L_seq, 6]
    X2_valid = np.stack(x2_valid_list).astype(np.float32)   # [N, L_seq]
    BW2 = np.stack(bw2_list).astype(np.float32)             # [N, 6, L_bw]
    BW2_valid = np.stack(bw2_valid_list).astype(np.float32) # [N, L_bw]

    # -------------------------
    # 调整输出格式
    # -------------------------
    X1 = maybe_transpose_seq(X1)   # [N, 6, L_seq] or [N, L_seq, 6]
    X2 = maybe_transpose_seq(X2)

    BW1 = maybe_transpose_bw(BW1)  # [N, 6, L_bw] or [N, L_bw, 6]
    BW2 = maybe_transpose_bw(BW2)

    # 标签与元信息
    y = subset["Label_num"].values.astype(np.float32)
    meta = subset["FromID"].values

    # -------------------------
    # 保存序列 npz
    # -------------------------
    seq_npz_path = os.path.join(seq_save_dir, f"{name}_seq.npz")
    np.savez_compressed(
        seq_npz_path,
        X1=X1,
        X2=X2,
        X1_valid=X1_valid,
        X2_valid=X2_valid,
        y=y,
        FromID=meta
    )

    # -------------------------
    # 保存 bigWig npz
    # -------------------------
    bw_npz_path = os.path.join(bw_save_dir, f"{name}_bw.npz")
    np.savez_compressed(
        bw_npz_path,
        BW1=BW1,
        BW2=BW2,
        BW1_valid=BW1_valid,
        BW2_valid=BW2_valid,
        y=y,
        FromID=meta,
        marks=np.array(list(bigwig_files.keys()))
    )

    print(f"\n{name} 集合保存完成:")
    print(f"  seq npz: {seq_npz_path}")
    print(f"  bw  npz: {bw_npz_path}")
    print("  X1 shape      :", X1.shape)
    print("  X2 shape      :", X2.shape)
    print("  X1_valid shape:", X1_valid.shape)
    print("  X2_valid shape:", X2_valid.shape)
    print("  BW1 shape     :", BW1.shape)
    print("  BW2 shape     :", BW2.shape)
    print("  BW1_valid     :", BW1_valid.shape)
    print("  BW2_valid     :", BW2_valid.shape)
    print("  y shape       :", y.shape)


# =========================================================
# 统计信息
# =========================================================
chr_assignments = []
for chr_name in df["chr1"].dropna().unique():
    if chr_name in train_chr:
        dataset = "train"
    elif chr_name in val_chr:
        dataset = "val"
    elif chr_name in test_chr:
        dataset = "test"
    else:
        dataset = "unassigned"

    subset_chr = df[df["chr1"] == chr_name]
    total = len(subset_chr)
    p_count = (subset_chr["Label"] == "P").sum()
    p_ratio = round(p_count / total, 4) if total > 0 else 0

    chr_assignments.append({
        "chr1": chr_name,
        "Dataset": dataset,
        "Total_samples": total,
        "P_samples": p_count,
        "P_ratio": p_ratio
    })

stats_df = pd.DataFrame(chr_assignments)

dataset_summary = []
for dataset_name, chroms in zip(
    ["train", "val", "test"],
    [train_chr, val_chr, test_chr]
):
    subset_ds = df[df["chr1"].isin(chroms)]
    total = len(subset_ds)
    p_count = (subset_ds["Label"] == "P").sum()
    p_ratio = round(p_count / total, 4) if total > 0 else 0

    dataset_summary.append({
        "Dataset": dataset_name,
        "Total_samples": total,
        "P_samples": p_count,
        "P_ratio": p_ratio,
        "Chromosomes": ",".join(chroms)
    })

summary_df = pd.DataFrame(dataset_summary)

stats_path = os.path.join(save_dir, "dataset_stats.xlsx")
with pd.ExcelWriter(stats_path) as writer:
    stats_df.to_excel(writer, sheet_name="chr1_stats", index=False)
    summary_df.to_excel(writer, sheet_name="dataset_summary", index=False)

print("\n统计信息已保存:", stats_path)


# =========================================================
# 保存 train / val / test
# =========================================================
process_subset(train_chr, "train")
process_subset(val_chr, "val")
process_subset(test_chr, "test")


# =========================================================
# 关闭 bigWig
# =========================================================
for mark, bw in bw_dict.items():
    bw.close()

print("\n数据预处理完成:", save_dir)