import streamlit as st
import torch
import numpy as np
import nibabel as nib
import tempfile
import os
import time
import pandas as pd
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════
#  Model Architecture (unchanged from UI.py)
# ══════════════════════════════════════════════════════════════════════

class Conv3dBnReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            mid = max(out_ch // reduction, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(out_ch, mid, bias=False), nn.ReLU(inplace=True),
                nn.Linear(mid, out_ch, bias=False), nn.Sigmoid(),
            )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            scale = self.se(out).view(out.size(0), out.size(1), 1, 1, 1)
            out   = out * scale
        return self.relu(out + self.shortcut(x))


class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256, use_se=True, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = ResBlock3D(32, 64, stride=2, use_se=use_se)
        self.layer2 = ResBlock3D(64, 128, stride=2, use_se=use_se)
        self.layer3 = ResBlock3D(128, 256, stride=2, use_se=use_se)
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj    = nn.Linear(256, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.proj(x)


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.block1 = self._res_block(in_dim, hidden_dim, dropout)
        self.block2 = self._res_block(hidden_dim, hidden_dim, dropout)
        self.proj   = nn.Linear(hidden_dim, embed_dim)
        self.sc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.sc2 = nn.Identity()

    @staticmethod
    def _res_block(in_d, out_d, drop):
        return nn.Sequential(
            nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(out_d, out_d), nn.LayerNorm(out_d), nn.GELU(),
        )

    def forward(self, x):
        h = self.block1(x) + self.sc1(x)
        h = self.block2(h) + h
        return self.proj(h)


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h, self.d_k = num_heads, embed_dim // num_heads
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        B = query.size(0)
        Q = self.W_Q(query).view(B, 1, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(context).view(B, 1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(context).view(B, 1, self.h, self.d_k).transpose(1, 2)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1)
        out = self.out(out)
        return self.norm(out + query)


class GatedMultimodalUnit(nn.Module):
    def __init__(self, embed_dim: int, n_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n = n_modalities
        self.gate_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_modalities)])
        self.feat_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_modalities)])
        self.gate_norm = nn.Linear(embed_dim * n_modalities, n_modalities)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, *embeddings):
        assert len(embeddings) == self.n
        z = [torch.tanh(self.gate_layers[i](e)) for i, e in enumerate(embeddings)]
        h = [torch.tanh(self.feat_layers[i](e)) for i, e in enumerate(embeddings)]
        gate_input = torch.cat(z, dim=-1)
        gates = F.softmax(self.gate_norm(gate_input), dim=-1)
        out = sum(gates[:, i:i+1] * h[i] for i in range(self.n))
        return self.norm(self.drop(out))


class TrimodalFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, out_dim=512, dropout=0.1):
        super().__init__()
        self.mri_pet = CrossModalAttention(embed_dim, num_heads, dropout)
        self.pet_mri = CrossModalAttention(embed_dim, num_heads, dropout)
        self.mri_tab = CrossModalAttention(embed_dim, num_heads, dropout)
        self.pet_tab = CrossModalAttention(embed_dim, num_heads, dropout)
        self.gmu = GatedMultimodalUnit(embed_dim, n_modalities=3, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, z_mri, z_pet, z_tab):
        z_mri_p  = self.mri_pet(z_mri, z_pet)
        z_pet_p  = self.pet_mri(z_pet, z_mri)
        z_mri_pp = self.mri_tab(z_mri_p, z_tab)
        z_pet_pp = self.pet_tab(z_pet_p, z_tab)
        z_fused  = self.gmu(z_mri_pp, z_pet_pp, z_tab)
        return self.proj(z_fused)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class AlzheimerMultimodalNet(nn.Module):
    def __init__(self, tab_in_dim=10, embed_dim=128, fused_dim=256, num_classes=2, dropout=0.3, use_se=True):
        super().__init__()
        self.mri_encoder = ResNet3D(in_channels=1, embed_dim=embed_dim, use_se=use_se, dropout=dropout)
        self.pet_encoder = ResNet3D(in_channels=1, embed_dim=embed_dim, use_se=use_se, dropout=dropout)
        self.tab_encoder = TabularEncoder(in_dim=tab_in_dim, hidden_dim=128, embed_dim=embed_dim, dropout=dropout)
        self.fusion = TrimodalFusion(embed_dim=embed_dim, num_heads=4, out_dim=fused_dim, dropout=dropout)
        self.classifier = ClassificationHead(in_dim=fused_dim, num_classes=num_classes, dropout=dropout + 0.1)
        self.aux_mri = nn.Linear(embed_dim, num_classes)
        self.aux_pet = nn.Linear(embed_dim, num_classes)
        self.aux_tab = nn.Linear(embed_dim, num_classes)

    def forward(self, mri, pet, tab, return_aux=False):
        z_mri = self.mri_encoder(mri)
        z_pet = self.pet_encoder(pet)
        z_tab = self.tab_encoder(tab)
        z_fused = self.fusion(z_mri, z_pet, z_tab)
        logits = self.classifier(z_fused)
        if return_aux:
            return logits, {"mri": self.aux_mri(z_mri), "pet": self.aux_pet(z_pet), "tab": self.aux_tab(z_tab)}
        return logits



#  Page Config

st.set_page_config(page_title="NeuroScan · AD Classifier", layout="wide", initial_sidebar_state="expanded")

TABULAR_COLS = [
    "AGE", "PTGENDER", "PTEDUCAT", "APOE4",
    "ABETA", "TAU", "PTAU",
    "ABETA_TAU_ratio", "ABETA_PTAU_ratio", "TAU_PTAU_ratio",
]

# ══════════════════════════════════════════════════════════════════════
#  Dark Theme CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:         #FFF8F0;
    --bg-raised:  #F5E6D3;
    --surface:    #FAEBD7;
    --surface-2:  #F0D9C0;
    --border:     #D4B896;
    --border-lt:  #E8CDB0;
    --glow:       rgba(139,90,43,0.10);
    --ink:        #2D1B0E;
    --ink-mid:    #4A3728;
    --ink-muted:  #7A6352;
    --ink-faint:  #A08B78;
    --accent:     #B8621B;
    --accent-2:   #9A4F10;
    --teal:       #1B7A6E;
    --teal-bg:    rgba(27,122,110,0.10);
    --teal-bdr:   rgba(27,122,110,0.30);
    --red:        #C0392B;
    --red-bg:     rgba(192,57,43,0.08);
    --red-bdr:    rgba(192,57,43,0.25);
    --green:      #1E8449;
    --green-bg:   rgba(30,132,73,0.08);
    --green-bdr:  rgba(30,132,73,0.25);
    --gold:       #D4A017;
    --gold-bg:    rgba(212,160,23,0.10);
    --purple:     #7D3C98;
    --violet:     #6C3483;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--ink) !important;
}
.stApp { background-color: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5E6D3 0%, #FAEBD7 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--ink-mid) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Card container */
.dark-card {
    background: linear-gradient(145deg, #FAEBD7, #F5E6D3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(139,90,43,0.10);
}
.dark-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--teal), transparent);
    opacity: 0.7;
}
.dark-card:hover { border-color: rgba(184,98,27,0.4); box-shadow: 0 6px 32px rgba(139,90,43,0.15); }

/* Hero */
.hero-block {
    background: linear-gradient(135deg, #F5E6D3 0%, #FAEBD7 30%, #FFF0E0 70%, #FFF8F0 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(139,90,43,0.12);
    text-align: center;
}
.hero-block::before {
    content: "";
    position: absolute;
    top: -50%; right: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(184,98,27,0.06), transparent 70%);
    pointer-events: none;
}
.hero-block::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--teal), var(--purple), var(--violet));
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem; color: #0E6B61;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 0.8rem; font-weight: 800;
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 3.5rem; color: #2D1B0E;
    margin: 0 0 0.5rem 0; font-weight: 800;
    line-height: 1.15; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #2D1B0E 0%, #9A4F10 50%, #1B7A6E 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--ink-mid); font-size: 1.1rem;
    font-weight: 500; margin: 0; letter-spacing: 0.3px;
}

/* Section labels */
.section-title {
    font-size: 1.35rem; color: var(--ink);
    font-weight: 700; display: flex;
    align-items: center; gap: 0.8rem;
    margin: 0; padding: 0;
}
.step-pill {
    background: linear-gradient(135deg, #E8740C, #F59E0B);
    color: #FFFFFF; border-radius: 8px;
    padding: 6px 16px; font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; letter-spacing: 1px;
    box-shadow: 0 2px 10px rgba(245,158,11,0.45);
}
.field-group-label {
    font-size: 0.75rem; font-weight: 700;
    color: var(--teal); text-transform: uppercase;
    letter-spacing: 1.5px; margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-lt);
}

/* Inputs */
label, .stTextInput label, .stNumberInput label, .stSlider label, .stSelectbox label {
    font-weight: 600 !important; color: var(--ink-mid) !important;
    font-size: 0.84rem !important;
}
input, [data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}
input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.2) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #A08B78 !important;
    border-radius: 12px !important;
    background: #6B5B4E !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #F59E0B !important;
    background: #5C4A3A !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.15) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    padding: 1.2rem 1.6rem !important;
}
[data-testid="stFileUploader"] small {
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button {
    background: #7A6A5C !important;
    color: #FFFFFF !important;
    border: 1px solid #A08B78 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #8C7A6B !important;
    color: #FFFFFF !important;
}
[data-testid="stFileUploaderFileName"] { color: #8C7A6B !important; font-weight: 600 !important; }

/* Main button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-2), var(--accent)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(154,79,16,0.35) !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 36px rgba(154,79,16,0.5) !important;
    transform: translateY(-2px);
}
.stButton > button:disabled {
    background: var(--surface-2) !important;
    color: var(--ink-faint) !important;
    box-shadow: none !important;
}

/* Readiness check */
.check-row {
    display: flex; align-items: center;
    gap: 0.7rem; padding: 0.6rem 0;
    font-size: 0.9rem; color: var(--ink-mid);
    border-bottom: 1px solid var(--border-lt);
}
.check-row:last-child { border-bottom: none; }
.check-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.check-dot.done { background: var(--green); box-shadow: 0 0 10px rgba(74,222,128,0.5); }
.check-dot.pending { background: var(--surface-2); border: 2px solid var(--ink-faint); }

/* Mini metric cards */
.mini-metric {
    background: linear-gradient(145deg, var(--surface), var(--surface-2));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem; text-align: center;
    box-shadow: 0 2px 12px rgba(139,90,43,0.08);
}
.mini-metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem; font-weight: 700; color: var(--accent);
}
.mini-metric-lbl {
    font-size: 0.64rem; color: var(--ink-muted);
    text-transform: uppercase; letter-spacing: 1.3px; margin-top: 4px;
}

/* Result cards */
.result-ad {
    background: var(--red-bg);
    border: 1px solid var(--red-bdr);
    border-left: 5px solid var(--red);
    border-radius: 14px; padding: 2rem 2.2rem;
    box-shadow: 0 4px 24px rgba(192,57,43,0.10);
}
.result-cn {
    background: var(--green-bg);
    border: 1px solid var(--green-bdr);
    border-left: 5px solid var(--green);
    border-radius: 14px; padding: 2rem 2.2rem;
    box-shadow: 0 4px 24px rgba(30,132,73,0.10);
}
.result-marker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;
}
.result-marker-ad { color: var(--red); }
.result-marker-cn { color: var(--green); }
.result-label {
    font-size: 2.2rem; margin: 0 0 0.4rem 0;
    line-height: 1.2; font-weight: 800;
}
.result-label-ad { color: var(--red); }
.result-label-cn { color: var(--green); }

/* Prob bars */
.prob-row { margin-bottom: 0.9rem; }
.prob-label {
    font-size: 0.85rem; color: var(--ink-mid);
    display: flex; justify-content: space-between;
    margin-bottom: 0.3rem; font-weight: 500;
}
.prob-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--ink-muted); }
.prob-bar-bg {
    background: var(--surface-2); border-radius: 6px;
    height: 12px; overflow: hidden;
}
.prob-bar-fill-ad {
    height: 100%; background: linear-gradient(90deg, #C0392B, #E74C3C, #F1948A);
    border-radius: 6px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-bar-fill-cn {
    height: 100%; background: linear-gradient(90deg, #1E8449, #27AE60, #82E0AA);
    border-radius: 6px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* Tips */
.info-tip {
    background: rgba(56,189,248,0.1);
    border-left: 3px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.2rem; font-size: 0.84rem;
    color: var(--accent); margin-top: 0.9rem; line-height: 1.55;
}
.warning-tip {
    background: var(--gold-bg);
    border-left: 3px solid var(--gold);
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.2rem; font-size: 0.84rem;
    color: var(--gold); margin-top: 0.8rem; line-height: 1.55;
}
.success-badge {
    background: var(--green-bg);
    border-left: 3px solid var(--green);
    border-radius: 0 12px 12px 0;
    padding: 0.7rem 1.2rem; font-size: 0.84rem;
    color: var(--green); font-weight: 600; margin-top: 0.6rem;
}
.csv-tag {
    display: inline-block;
    background: var(--teal-bg); border: 1px solid var(--teal-bdr);
    border-radius: 6px; padding: 4px 12px;
    font-size: 0.74rem; font-family: 'JetBrains Mono', monospace;
    color: var(--teal); margin: 3px; font-weight: 500;
}

/* Divider / Footer */
hr { border: none; border-top: 1px solid var(--border-lt); margin: 1.2rem 0; }

/* Selectbox — closed state */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: var(--ink) !important;
    background-color: var(--surface) !important;
}
/* Selectbox — dropdown / popover menu */
[data-baseweb="popover"] { background-color: var(--bg-raised) !important; border: 1px solid var(--border) !important; }
[data-baseweb="popover"] > div { background-color: var(--bg-raised) !important; }
[data-baseweb="menu"] { background-color: var(--bg-raised) !important; }
[data-baseweb="menu"] ul { background-color: var(--bg-raised) !important; }
[data-baseweb="menu"] li,
[role="option"] {
    color: var(--ink) !important;
    background-color: var(--bg-raised) !important;
}
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[data-baseweb="menu"] li[aria-selected="true"],
[role="option"][aria-selected="true"] {
    background-color: var(--surface) !important;
    color: var(--accent) !important;
}
ul[role="listbox"] { background-color: var(--bg-raised) !important; }
ul[role="listbox"] li { background-color: var(--bg-raised) !important; color: var(--ink) !important; }
ul[role="listbox"] li:hover { background-color: var(--surface) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] p {
    color: var(--ink-mid) !important;
}

/* Dataframe / Table */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] * {
    color: var(--ink) !important;
    background-color: transparent !important;
}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    background: var(--surface) !important;
}
.dvn-scroller { background: var(--surface) !important; }
.gdg-cell, .gdg-header, .gdg-header-cell {
    background: var(--surface) !important;
    color: var(--ink) !important;
}
iframe[title="streamlit_dataframe"] { background: var(--surface) !important; }

/* Number input buttons */
[data-testid="stNumberInput"] button {
    background: var(--surface-2) !important;
    color: var(--ink-mid) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--border) !important;
    color: var(--ink) !important;
}

/* Toast / alerts */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
}

/* Generic text */
.stMarkdown, .stText, .stCaption,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    color: var(--ink-mid) !important;
}

/* Browse files button */
[data-testid="stFileUploaderDropzoneInput"] + div button,
[data-testid="baseButton-secondary"] {
    background: var(--surface-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: var(--border) !important;
    color: var(--ink) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: var(--surface-2) !important;
    color: var(--ink-mid) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] button:hover {
    background: var(--border) !important;
}

/* Spinner */
.stSpinner > div { color: var(--ink-muted) !important; }

/* Streamlit header bar — dark */
[data-testid="stHeader"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border-lt) !important;
}
[data-testid="stHeader"] * { color: var(--ink-mid) !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stToolbar"] button { color: var(--ink-muted) !important; }

/* Ensure all text is dark on light background */
p, span, div, li, td, th { color: var(--ink) !important; }

.sidebar-title {
    font-size: 1.05rem; font-weight: 700;
    color: var(--ink) !important;
    letter-spacing: 0.2px; margin-bottom: 0.4rem;
}
.sidebar-status {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 0.6rem 0.85rem;
    font-size: 0.8rem; color: var(--ink-muted) !important;
    margin-top: 0.8rem; line-height: 1.5;
}
.sidebar-status.ready {
    background: var(--green-bg);
    border-color: var(--green-bdr);
    color: var(--green) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_path: str):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if hasattr(checkpoint, "eval"):
            model = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model = AlzheimerMultimodalNet(tab_in_dim=10)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = AlzheimerMultimodalNet(tab_in_dim=10)
            model.load_state_dict(checkpoint)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_nifti(file_bytes, filename: str, mode: str = "mri") -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        target = (96, 96, 96)
        from scipy.ndimage import zoom
        current_shape = data.shape[:3]
        if current_shape != tuple(target):
            factors = [t / s for t, s in zip(target, current_shape)]
            data = zoom(data, factors, order=1)
        if mode == "mri":
            mn, mx = data.min(), data.max()
            if mx - mn > 1e-8:
                data = (data - mn) / (mx - mn)
            data = np.clip(data, 0.0, 1.0)
        else:
            data = np.clip(data, 0.0, None)
        return data.astype(np.float32), img.shape
    finally:
        os.unlink(tmp_path)


def run_inference(model, mri_arr, pet_arr, tabular_arr):
    mri_t = torch.tensor(mri_arr).unsqueeze(0).unsqueeze(0)
    pet_t = torch.tensor(pet_arr).unsqueeze(0).unsqueeze(0)
    tab_t = torch.tensor(tabular_arr, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        try:
            logits = model(mri_t, pet_t, tab_t)
        except TypeError:
            combined = torch.cat([mri_t.flatten(1), pet_t.flatten(1), tab_t], dim=1)
            logits = model(combined)
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_class = int(np.argmax(probs))
    return pred_class, probs


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Model Upload
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title"> Model Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")
    model_file = st.file_uploader("Trained model (.pt / .pth)", type=["pt", "pth"], key="model_upload")
    model, model_err = None, None
    if model_file:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(model_file.read())
            tmp_path = tmp.name
        model, model_err = load_model(tmp_path)
        if model:
            st.markdown('<div class="sidebar-status ready">✓ Model loaded and ready</div>', unsafe_allow_html=True)
        else:
            st.error(f"Could not load model: {model_err}")
    else:
        st.markdown('<div class="sidebar-status">Upload a <code>model.pt</code> file to begin.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("NeuroScan · AD Classifier")


# ══════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — Vertical Stacked Design
# ══════════════════════════════════════════════════════════════════════

# ── Hero Header ──
st.markdown("""
<div class="hero-block">
  <div class="hero-eyebrow">Multimodal Neural Imaging &nbsp;•&nbsp; Research Use Only</div>
  <h1 class="hero-title">Alzheimer's Disease Stage Classification</h1>
  <p class="hero-sub">AD vs MCI &mdash; MRI + PET + CSF Biomarkers + Clinical Scores</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════  SECTION 1: MRI Upload  ════════════════════════
st.markdown("""
<div class="dark-card">
<div class="section-title"><span class="step-pill">01</span> MRI Scan Upload</div>
</div>
""", unsafe_allow_html=True)

mri_file = st.file_uploader("Upload preprocessed MRI (.nii.gz)", type=["gz", "nii"], key="mri", label_visibility="collapsed")
mri_info = None
if mri_file:
    with st.spinner("Reading MRI volume..."):
        mri_arr, mri_shape = preprocess_nifti(mri_file.read(), mri_file.name)
    mri_info = mri_arr
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{mri_shape[0]}</div><div class="mini-metric-lbl">X dim</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{mri_shape[1]}</div><div class="mini-metric-lbl">Y dim</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{mri_shape[2]}</div><div class="mini-metric-lbl">Z dim</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mini-metric"><div class="mini-metric-val">96³</div><div class="mini-metric-lbl">Resampled</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="success-badge">✓ MRI volume loaded successfully</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════  SECTION 2: PET Upload  ════════════════════════
st.markdown("""
<div class="dark-card">
<div class="section-title"><span class="step-pill">02</span> PET Scan Upload</div>
</div>
""", unsafe_allow_html=True)

pet_file = st.file_uploader("Upload preprocessed PET (.nii.gz)", type=["gz", "nii"], key="pet", label_visibility="collapsed")
pet_info = None
if pet_file:
    with st.spinner("Reading PET volume..."):
        pet_arr, pet_shape = preprocess_nifti(pet_file.read(), pet_file.name, mode="pet")
    pet_info = pet_arr
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{pet_shape[0]}</div><div class="mini-metric-lbl">X dim</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{pet_shape[1]}</div><div class="mini-metric-lbl">Y dim</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{pet_shape[2]}</div><div class="mini-metric-lbl">Z dim</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mini-metric"><div class="mini-metric-val">96³</div><div class="mini-metric-lbl">Resampled</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="success-badge">✓ PET volume loaded successfully</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════  SECTION 3: Clinical Data  ═════════════════════
st.markdown("""
<div class="dark-card">
<div class="section-title"><span class="step-pill">03</span> Clinical &amp; Biomarker Data</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-tip">
Upload a CSV file to auto-fill patient data, or enter values manually below. CSV headers are matched automatically.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# CSV Auto-fill
csv_upload = st.file_uploader("Upload CSV for auto-fill (optional)", type=["csv"], key="csv_autofill", label_visibility="collapsed")

csv_data = None
selected_ptid = None
if csv_upload:
    try:
        csv_data = pd.read_csv(csv_upload)
        # Normalize column names
        csv_data.columns = [c.strip() for c in csv_data.columns]
        st.markdown('<div class="success-badge">✓ CSV loaded — columns detected:</div>', unsafe_allow_html=True)
        tags_html = " ".join([f'<span class="csv-tag">{c}</span>' for c in csv_data.columns])
        st.markdown(f'<div style="margin:0.5rem 0 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

        # Patient selector
        if "PTID" in csv_data.columns:
            id_col = "PTID"
        elif "patient_id" in csv_data.columns:
            id_col = "patient_id"
        else:
            id_col = None

        if id_col:
            selected_ptid = st.selectbox(f"Select Patient ({id_col})", csv_data[id_col].tolist())
            row = csv_data[csv_data[id_col] == selected_ptid].iloc[0]
        else:
            row_idx = st.number_input("Select Row Index", min_value=0, max_value=len(csv_data)-1, value=0, step=1)
            row = csv_data.iloc[int(row_idx)]

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        csv_data = None

# Helper to get value from CSV row or use default
def csv_val(col, default, row=None):
    if csv_data is not None and row is not None and col in csv_data.columns:
        try:
            return float(row[col])
        except (ValueError, TypeError):
            return default
    return default

def csv_gender(row=None):
    if csv_data is not None and row is not None and "PTGENDER" in csv_data.columns:
        val = str(row["PTGENDER"]).strip().lower()
        if val in ["male", "1", "1.0"]:
            return 1
        elif val in ["female", "2", "2.0"]:
            return 2
    return 1

# Get values (either from CSV or defaults)
_row = row if csv_data is not None else None

# ── Demographics ──
st.markdown('<p class="field-group-label">Demographics</p>', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
with d1:
    age = st.number_input("AGE", min_value=40.0, max_value=100.0, value=csv_val("AGE", 72.0, _row), step=0.5)
with d2:
    ptgender = st.selectbox("PTGENDER", options=[1, 2], index=0 if csv_gender(_row) == 1 else 1,
                            format_func=lambda x: "Male (1)" if x == 1 else "Female (2)")
with d3:
    pteducat = st.number_input("PTEDUCAT", min_value=0.0, max_value=30.0, value=csv_val("PTEDUCAT", 16.0, _row), step=1.0)

# ── Genetic Risk ──
st.markdown('<p class="field-group-label">Genetic Risk</p>', unsafe_allow_html=True)
apoe4_val = int(csv_val("APOE4", 0, _row))
apoe4 = st.selectbox("APOE4 alleles", options=[0, 1, 2], index=min(apoe4_val, 2),
                      format_func=lambda x: f"{x} allele{'s' if x != 1 else ''}")

# ── CSF Biomarkers ──
st.markdown('<p class="field-group-label">CSF Biomarkers</p>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    abeta = st.number_input("ABETA", min_value=0.0, value=csv_val("ABETA", 150.0, _row), step=1.0)
with b2:
    tau = st.number_input("TAU", min_value=0.0, value=csv_val("TAU", 80.0, _row), step=1.0)
with b3:
    ptau = st.number_input("PTAU", min_value=0.0, value=csv_val("PTAU", 25.0, _row), step=0.5)

# ── Derived Ratios ──
st.markdown('<p class="field-group-label">Derived Ratios <span style="font-weight:400;color:var(--ink-faint);text-transform:none;letter-spacing:0;">(auto-computed)</span></p>', unsafe_allow_html=True)
abeta_tau_ratio  = abeta / tau   if tau  > 0 else 0.0
abeta_ptau_ratio = abeta / ptau  if ptau > 0 else 0.0
tau_ptau_ratio   = tau   / ptau  if ptau > 0 else 0.0

r1, r2, r3 = st.columns(3)
r1.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{abeta_tau_ratio:.2f}</div><div class="mini-metric-lbl">ABETA / TAU</div></div>', unsafe_allow_html=True)
r2.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{abeta_ptau_ratio:.2f}</div><div class="mini-metric-lbl">ABETA / PTAU</div></div>', unsafe_allow_html=True)
r3.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{tau_ptau_ratio:.2f}</div><div class="mini-metric-lbl">TAU / PTAU</div></div>', unsafe_allow_html=True)

# Assemble tabular array
input_values = [
    float(age), float(ptgender), float(pteducat), float(apoe4),
    float(abeta), float(tau), float(ptau),
    abeta_tau_ratio, abeta_ptau_ratio, tau_ptau_ratio,
]
tabular_data = np.array(input_values, dtype=np.float32)
csv_df = pd.DataFrame([dict(zip(TABULAR_COLS, input_values))])


st.markdown('<div class="success-badge">✓ Clinical data ready</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════  SECTION 4: Run Prediction  ════════════════════
st.markdown("""
<div class="dark-card">
<div class="section-title"><span class="step-pill">04</span> Run Prediction</div>
</div>
""", unsafe_allow_html=True)

# Readiness checklist
checks = {
    "Model loaded":        model is not None,
    "MRI uploaded":        mri_info is not None,
    "PET uploaded":        pet_info is not None,
    "Clinical data ready": tabular_data is not None,
}

st.markdown('<div style="margin-bottom:1.2rem;">', unsafe_allow_html=True)
for label, ok in checks.items():
    dot_class = "done" if ok else "pending"
    st.markdown(f'<div class="check-row"><div class="check-dot {dot_class}"></div>{label}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

ready = all(checks.values())
predict_btn = st.button("Run AD Classification", disabled=not ready)

if not ready:
    st.markdown('<div class="warning-tip">Complete all steps above to enable prediction.</div>', unsafe_allow_html=True)

# ── Prediction Result ──
if predict_btn and ready:
    with st.spinner("Analysing multimodal data..."):
        time.sleep(0.6)
        pred_class, probs = run_inference(model, mri_info, pet_info, tabular_data)

    labels     = ["MCI", "AD"]
    label      = labels[pred_class]
    confidence = float(probs[pred_class]) * 100
    prob_ad    = float(probs[1]) * 100
    prob_cn    = float(probs[0]) * 100

    st.markdown("<hr>", unsafe_allow_html=True)

    if label == "AD":
        st.markdown(f"""
        <div class="result-ad">
          <div class="result-marker result-marker-ad">Classification Result</div>
          <div class="result-label result-label-ad">Alzheimer Disease</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-cn">
          <div class="result-marker result-marker-cn">Classification Result</div>
          <div class="result-label result-label-cn">Mild Cognitive Impairment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

