import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 64
NUM_CLASSES = 4
DROPOUT_P   = 0.3
ROUTING_ITERS = 3
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MODEL_PATH  = 'fuzzy_capsnet.pt'

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
#  MODEL DEFINITION
# ─────────────────────────────────────────────
def squash(x, dim=-1):
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm    = norm_sq.sqrt()
    scale   = norm_sq / (1.0 + norm_sq)
    return scale * (x / (norm + 1e-8))


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, dropout_p=DROPOUT_P):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.dropout(self.block3(self.block2(self.block1(x))))


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels=128, capsule_dim=8, num_capsules=32):
        super().__init__()
        self.capsule_dim  = capsule_dim
        self.num_capsules = num_capsules
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        B, _, H, W = out.shape
        out = out.view(B, self.num_capsules, self.capsule_dim, H * W)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, -1, self.capsule_dim)
        return squash(out)


class FuzzyCapsuleLayer(nn.Module):
    def __init__(self, num_in_capsules, num_out_capsules=NUM_CLASSES,
                 in_dim=8, out_dim=16, routing_iters=ROUTING_ITERS):
        super().__init__()
        self.num_in  = num_in_capsules
        self.num_out = num_out_capsules
        self.iters   = routing_iters
        self.W = nn.Parameter(
            torch.randn(1, num_in_capsules, num_out_capsules, out_dim, in_dim) * 0.01)
        self.log_sigma = nn.Parameter(torch.ones(num_out_capsules) * 0.5)

    def forward(self, u):
        B = u.size(0)
        u_exp = u.unsqueeze(2).unsqueeze(-1)
        W_exp = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W_exp, u_exp).squeeze(-1)
        b = torch.zeros(B, self.num_in, self.num_out, device=u.device, dtype=u.dtype)
        v = None
        for iteration in range(self.iters):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)
            if iteration == self.iters - 1:
                break
            sigma    = self.log_sigma.exp().clamp(min=1e-4)
            v_exp    = v.unsqueeze(1).expand_as(u_hat)
            diff_sq  = ((u_hat - v_exp) ** 2).sum(dim=-1)
            sigma_sq = (sigma ** 2).unsqueeze(0).unsqueeze(0)
            membership = torch.exp(-diff_sq / (2 * sigma_sq + 1e-8))
            b = b + membership
        return v


class FuzzyCapsNet(nn.Module):
    def __init__(self, img_size=IMG_SIZE, in_channels=3,
                 num_classes=NUM_CLASSES, dropout_p=DROPOUT_P,
                 routing_iters=ROUTING_ITERS):
        super().__init__()
        feat_spatial = img_size // 8
        num_primary  = 32 * (feat_spatial ** 2)
        self.conv_extractor = ConvFeatureExtractor(in_channels, dropout_p)
        self.primary_caps   = PrimaryCapsLayer(128, 8, 32)
        self.fuzzy_caps     = FuzzyCapsuleLayer(num_primary, num_classes, 8, 16, routing_iters)
        self.dropout        = nn.Dropout(p=dropout_p)
        self.num_classes    = num_classes

    def forward(self, x):
        features = self.conv_extractor(x)
        u        = self.primary_caps(features)
        u        = self.dropout(u)
        v        = self.fuzzy_caps(u)
        norms    = v.norm(dim=-1)
        return norms, norms

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            scores, norms = self(x)
        probs      = F.softmax(norms, dim=-1)
        norm_min   = norms.min(dim=-1, keepdim=True).values
        norm_max   = norms.max(dim=-1, keepdim=True).values
        membership = (norms - norm_min) / (norm_max - norm_min + 1e-8)
        return probs, membership


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
device = torch.device('cpu')
model  = FuzzyCapsNet(img_size=IMG_SIZE).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: {MODEL_PATH} not found — running with random weights.")

# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────
BG        = '#050C14'
PANEL     = '#071424'
BORDER    = '#0A2A4A'
CYAN      = '#00D4FF'
CYAN_DIM  = '#0099BB'
GREEN     = '#00FF88'
YELLOW    = '#FFD600'
RED       = '#FF4466'
TEXT      = '#C8E8F8'
TEXT_DIM  = '#4A7A9B'

CLASS_COLORS = {
    'Glioma':      RED,
    'Meningioma':  YELLOW,
    'No Tumor':    GREEN,
    'Pituitary':   CYAN,
}

CLASS_INFO = {
    'Glioma': {
        'risk': 'HIGH',
        'risk_color': RED,
        'description': 'Tumor originating from glial cells. Most common primary brain tumor.',
        'location': 'Cerebral hemispheres',
        'note': 'Immediate specialist referral recommended.',
    },
    'Meningioma': {
        'risk': 'MODERATE',
        'risk_color': YELLOW,
        'description': 'Arises from meninges surrounding the brain and spinal cord.',
        'location': 'Brain surface / skull base',
        'note': 'Usually benign. Monitor or surgical evaluation.',
    },
    'No Tumor': {
        'risk': 'NONE',
        'risk_color': GREEN,
        'description': 'No tumor detected in the MRI scan.',
        'location': 'N/A',
        'note': 'No abnormalities detected.',
    },
    'Pituitary': {
        'risk': 'LOW–MOD',
        'risk_color': CYAN,
        'description': 'Tumor located in the pituitary gland at the base of the brain.',
        'location': 'Pituitary gland',
        'note': 'Hormonal assessment recommended.',
    },
}

# ─────────────────────────────────────────────
#  INFERENCE + PLOT
# ─────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, build_empty_plot(), build_empty_membership(), build_model_info()

    # Convert & run
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    probs_t, memb_t = model.predict_proba(tensor)
    probs      = probs_t[0].numpy()
    membership = memb_t[0].numpy()
    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    info       = CLASS_INFO[pred_class]

    result_plot   = build_result_plot(probs, membership, pred_idx, pred_class, confidence, info)
    membership_fig = build_membership_chart(membership, pred_idx)
    model_fig     = build_model_info()

    return result_plot, membership_fig, model_fig


def styled_ax(ax, facecolor=PANEL):
    ax.set_facecolor(facecolor)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT)


def build_result_plot(probs, membership, pred_idx, pred_class, confidence, info):
    fig = plt.figure(figsize=(16, 5.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                            left=0.05, right=0.97, top=0.84, bottom=0.16)

    accent = CLASS_COLORS[pred_class]

    # ── Panel 1: Softmax bar chart ──
    ax1 = fig.add_subplot(gs[0])
    styled_ax(ax1)
    ax1.set_title('SOFTMAX PROBABILITIES', color=TEXT_DIM, fontsize=11,
                  fontfamily='monospace', pad=10, loc='left')
    colors = [accent if i == pred_idx else BORDER for i in range(NUM_CLASSES)]
    bars   = ax1.barh(CLASS_NAMES, probs * 100, color=colors,
                      height=0.55, edgecolor='none')
    ax1.set_xlim(0, 120)
    ax1.set_xlabel('Confidence (%)', fontsize=11, color=TEXT_DIM)
    ax1.invert_yaxis()
    for bar, val, name in zip(bars, probs * 100, CLASS_NAMES):
        c = accent if CLASS_NAMES.index(name) == pred_idx else TEXT_DIM
        ax1.text(val + 2, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=12,
                 color=c, fontfamily='monospace')
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_yticklabels(CLASS_NAMES, fontsize=12, fontfamily='monospace', color=TEXT)
    ax1.tick_params(colors=TEXT_DIM, labelsize=12)
    ax1.grid(axis='x', color=BORDER, linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    # ── Panel 2: Capsule norms radar-style (polar) ──
    ax2 = fig.add_subplot(gs[1], polar=True)
    ax2.set_facecolor(PANEL)
    angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False).tolist()
    vals   = membership.tolist()
    angles_closed = angles + [angles[0]]
    vals_closed   = vals   + [vals[0]]
    ax2.plot(angles_closed, vals_closed, color=accent, linewidth=2, alpha=0.9)
    ax2.fill(angles_closed, vals_closed, color=accent, alpha=0.15)
    ax2.scatter(angles, vals, color=accent, s=60, zorder=5)
    ax2.set_xticks(angles)
    ax2.set_xticklabels(CLASS_NAMES, color=TEXT, fontsize=11, fontfamily='monospace')
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(['0.25', '0.5', '0.75', '1.0'],
                        color=TEXT_DIM, fontsize=9, fontfamily='monospace')
    ax2.set_ylim(0, 1)
    ax2.grid(color=BORDER, linewidth=0.6)
    ax2.spines['polar'].set_color(BORDER)
    ax2.set_title('CAPSULE ROUTING MAP', color=TEXT_DIM,
                  fontsize=11, fontfamily='monospace', pad=16, loc='center')

    # ── Panel 3: Prediction summary ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PANEL)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis('off')

    rect = plt.Rectangle((0.03, 0.05), 0.94, 0.9,
                          linewidth=1.2, edgecolor=accent, facecolor='none', alpha=0.5)
    ax3.add_patch(rect)
    for x, y, dx, dy in [(0.03, 0.95, 0.14, 0), (0.03, 0.95, 0, -0.14),
                          (0.97, 0.05, -0.14, 0), (0.97, 0.05, 0, 0.14)]:
        ax3.plot([x, x+dx], [y, y+dy], color=accent, linewidth=2.5)

    ax3.text(0.5, 0.88, 'DIAGNOSIS', ha='center', va='center',
             color=TEXT_DIM, fontsize=11, fontfamily='monospace')
    ax3.text(0.5, 0.72, pred_class.upper(), ha='center', va='center',
             color=accent, fontsize=20, fontfamily='monospace', fontweight='bold')
    ax3.axhline(0.62, xmin=0.1, xmax=0.9, color=BORDER, linewidth=0.8)

    ax3.text(0.5, 0.52, f'{confidence:.1f}%', ha='center', va='center',
             color=TEXT, fontsize=28, fontfamily='monospace', fontweight='bold')
    ax3.text(0.5, 0.41, 'CONFIDENCE', ha='center', va='center',
             color=TEXT_DIM, fontsize=11, fontfamily='monospace')
    ax3.axhline(0.33, xmin=0.1, xmax=0.9, color=BORDER, linewidth=0.8)

    risk_c = info['risk_color']
    ax3.text(0.12, 0.23, 'RISK LEVEL', ha='left', va='center',
             color=TEXT_DIM, fontsize=10, fontfamily='monospace')
    ax3.text(0.88, 0.23, info['risk'], ha='right', va='center',
             color=risk_c, fontsize=12, fontfamily='monospace', fontweight='bold')

    ax3.text(0.12, 0.13, 'LOCATION', ha='left', va='center',
             color=TEXT_DIM, fontsize=10, fontfamily='monospace')
    ax3.text(0.88, 0.13, info['location'], ha='right', va='center',
             color=TEXT, fontsize=10, fontfamily='monospace')

    fig.suptitle(
        f'Fuzzy-CapsNet  |  Brain MRI Analysis  —  {pred_class}  detected',
        color=TEXT, fontsize=13, fontfamily='monospace', y=0.97)

    return fig


def build_membership_chart(membership, pred_idx):
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    styled_ax(ax)
    ax.set_title('FUZZY MEMBERSHIP SCORES  —  Gaussian Routing Agreement',
                 color=TEXT_DIM, fontsize=11, fontfamily='monospace', pad=10, loc='left')

    accent = CLASS_COLORS[CLASS_NAMES[pred_idx]]
    colors = [accent if i == pred_idx else '#1A3A5C' for i in range(NUM_CLASSES)]

    bars = ax.bar(CLASS_NAMES, membership, color=colors, width=0.5,
                  edgecolor=BG, linewidth=1.5)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('μ  (0 = non-member,  1 = full member)', fontsize=11, color=TEXT_DIM)

    for bar, val, i in zip(bars, membership, range(NUM_CLASSES)):
        c = accent if i == pred_idx else TEXT_DIM
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=12, color=c, fontfamily='monospace')

    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontfamily='monospace', color=TEXT)
    ax.tick_params(colors=TEXT_DIM, labelsize=12)
    ax.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def build_model_info():
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    ax.text(0.02, 0.95, 'MODEL  ARCHITECTURE  SUMMARY',
            color=CYAN_DIM, fontsize=11, fontfamily='monospace', va='top', fontweight='bold')
    ax.axhline(0.87, xmin=0.02, xmax=0.98, color=BORDER, linewidth=0.8)

    specs = [
        ('Architecture',   'Fuzzy Capsule Network  (Fuzzy-CapsNet)'),
        ('Backbone',       'ConvFeatureExtractor  — 3-block CNN  [32→64→128]'),
        ('Capsule Layers', 'PrimaryCaps (8-D)  →  FuzzyCaps (16-D)'),
        ('Routing',        'Gaussian Membership  —  3 iters  —  σ learnable'),
        ('Loss',           'CapsNet Margin Loss  (Hinton et al., 2017)'),
        ('Parameters',     '1,437,444  trainable'),
        ('Input',          '64 × 64 × 3  (RGB MRI scan)'),
        ('Test Accuracy',  '91.50%     ROC-AUC : 96.47%'),
    ]

    y = 0.80
    for label, value in specs:
        ax.text(0.02, y, f'{label:<16}', color=TEXT_DIM,
                fontsize=10, fontfamily='monospace', va='top')
        ax.text(0.30, y, value, color=TEXT,
                fontsize=10, fontfamily='monospace', va='top')
        y -= 0.096

    plt.tight_layout()
    return fig


def build_empty_plot():
    fig, ax = plt.subplots(figsize=(16, 5.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.55, 'Upload an MRI scan to begin analysis',
            ha='center', va='center', color=TEXT_DIM,
            fontsize=15, fontfamily='monospace')
    ax.text(0.5, 0.40, '[ awaiting input ]',
            ha='center', va='center', color=BORDER,
            fontsize=12, fontfamily='monospace')
    return fig


def build_empty_membership():
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.axis('off')
    ax.text(0.5, 0.5, '—', ha='center', va='center',
            color=BORDER, fontsize=20, fontfamily='monospace')
    return fig


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

body, .gradio-container {
    background-color: #050C14 !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: #C8E8F8 !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* Header */
.app-header {
    background: linear-gradient(180deg, #071424 0%, #050C14 100%);
    border-bottom: 1px solid #0A2A4A;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}
.app-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 18px;
    color: #00D4FF;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.app-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #4A7A9B;
    letter-spacing: 2px;
}
.app-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #00FF88;
    border: 1px solid #00FF88;
    padding: 3px 8px;
    letter-spacing: 1px;
}

/* Panels */
.panel-label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 9px !important;
    color: #4A7A9B !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* Upload area */
.upload-box .wrap {
    background: #071424 !important;
    border: 1px solid #0A2A4A !important;
    border-radius: 0px !important;
    min-height: 180px !important;
}
.upload-box .wrap:hover {
    border-color: #00D4FF !important;
}
.upload-box .icon-wrap { color: #0A2A4A !important; }
.upload-box .upload-text { color: #4A7A9B !important; font-family: 'Share Tech Mono', monospace !important; font-size: 11px !important; }

/* Button */
.analyze-btn {
    background: transparent !important;
    border: 1px solid #00D4FF !important;
    color: #00D4FF !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    border-radius: 0 !important;
    padding: 10px 0 !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.analyze-btn:hover {
    background: #00D4FF22 !important;
    box-shadow: 0 0 12px #00D4FF44 !important;
}

/* Plot outputs */
.plot-output {
    background: #071424 !important;
    border: 1px solid #0A2A4A !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.plot-output canvas { border-radius: 0 !important; }

/* Info panel */
.info-box {
    background: #071424;
    border: 1px solid #0A2A4A;
    padding: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #4A7A9B;
    line-height: 1.9;
}
.info-box b { color: #00D4FF; }

/* Footer */
.app-footer {
    border-top: 1px solid #0A2A4A;
    padding: 10px 0;
    margin-top: 14px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #1A3A5C;
    letter-spacing: 1px;
}

/* Hide Gradio branding */
footer { display: none !important; }
.svelte-1ipelgc { display: none !important; }
"""

# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Fuzzy-CapsNet | Brain MRI Classifier") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <div>
            <div class="app-title">Fuzzy-CapsNet</div>
            <div class="app-subtitle">Brain MRI Tumor Classification System</div>
        </div>
        <div class="app-badge">Model v1.0 &nbsp;|&nbsp; 91.50% Test Acc</div>
    </div>
    """)

    # ── Main layout ──
    with gr.Row():

        # Left column — upload + info
        with gr.Column(scale=1, min_width=200):
            gr.HTML('<div class="panel-label">MRI Input</div>')
            image_input = gr.Image(
                type='pil',
                label='',
                elem_classes='upload-box',
                show_label=False,
                height=220,
            )
            analyze_btn = gr.Button(
                "RUN ANALYSIS",
                elem_classes='analyze-btn',
            )
            gr.HTML("""
            <div class="info-box" style="margin-top:10px; font-size:11px; line-height:1.8">
                <b>SUPPORTED CLASSES</b><br>
                &nbsp;01 &nbsp;Glioma<br>
                &nbsp;02 &nbsp;Meningioma<br>
                &nbsp;03 &nbsp;No Tumor<br>
                &nbsp;04 &nbsp;Pituitary<br><br>
                <b>INPUT FORMAT</b><br>
                &nbsp;PNG / JPG / JPEG<br>
                &nbsp;Auto-resized to 64x64<br><br>
                <b>NOTE</b><br>
                &nbsp;Research use only.<br>
                &nbsp;Not a clinical tool.
            </div>
            """)

        # Right column — results
        with gr.Column(scale=4):
            gr.HTML('<div class="panel-label">Analysis Output</div>')
            result_plot = gr.Plot(
                label='',
                show_label=False,
                elem_classes='plot-output',
            )

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="panel-label" style="margin-top:12px">Fuzzy Membership Scores</div>')
                    membership_plot = gr.Plot(
                        label='',
                        show_label=False,
                        elem_classes='plot-output',
                    )
                with gr.Column():
                    gr.HTML('<div class="panel-label" style="margin-top:12px">Model Architecture</div>')
                    model_plot = gr.Plot(
                        label='',
                        show_label=False,
                        elem_classes='plot-output',
                    )

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
        Fuzzy-CapsNet &nbsp;|&nbsp; Hybrid Fuzzy Capsule Network &nbsp;|&nbsp;
        Gaussian Membership Routing &nbsp;|&nbsp; PyTorch &nbsp;|&nbsp;
        For research purposes only
    </div>
    """)

    # ── Wire up ──
    analyze_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[result_plot, membership_plot, model_plot],
    )
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[result_plot, membership_plot, model_plot],
    )

    # Load defaults on startup
    demo.load(
        fn=lambda: (build_empty_plot(), build_empty_membership(), build_model_info()),
        outputs=[result_plot, membership_plot, model_plot],
    )

if __name__ == '__main__':
    demo.launch()