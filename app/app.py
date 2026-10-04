from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
EXPECTED_CHECKPOINT = MODEL_DIR / "vgg16_byol_finetuned_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="SpineSight MRI | Lumbar MRI Classification",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #f5f8fc; color: #14213d;}
.block-container {max-width: 1480px; padding-top: 0.7rem; padding-bottom: 1.5rem;}
html, body, [class*="css"] {font-family: "Segoe UI", Arial, sans-serif;}

.hero {
  background: linear-gradient(135deg, #0b1f3a 0%, #153d67 55%, #1b6a8a 100%);
  border-radius: 22px;
  padding: 1.15rem 1.35rem;
  color: white;
  box-shadow: 0 10px 30px rgba(13, 42, 72, 0.16);
  margin-bottom: 0.85rem;
}
.hero-title {font-size: 2.05rem; line-height: 1.05; font-weight: 900; margin: 0;}
.hero-sub {font-size: 1.02rem; margin: .42rem 0 0; color: #d9efff; font-weight: 500;}
.hero-badge {
  display: inline-block; margin-top: .62rem; padding: .28rem .62rem;
  background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.26);
  border-radius: 999px; font-weight: 800; font-size: .82rem;
}

.card {
  background: #ffffff;
  border: 1px solid #dce6f1;
  border-radius: 18px;
  padding: 1rem 1.05rem;
  box-shadow: 0 6px 20px rgba(22, 56, 88, 0.07);
  margin-bottom: .85rem;
}
.section-label {
  text-transform: uppercase; letter-spacing: .09em; font-size: .77rem;
  font-weight: 900; color: #476985; margin-bottom: .36rem;
}
.section-title {font-size: 1.25rem; font-weight: 900; color: #102f4f; margin: 0 0 .48rem;}

.metric-card {
  background: #ffffff;
  border: 1px solid #d9e5ef;
  border-radius: 16px;
  padding: .76rem .9rem;
  min-height: 104px;
  box-shadow: 0 4px 16px rgba(22, 56, 88, .06);
}
.metric-name {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 900; color: #5c748b;}
.metric-value {font-size: 1.75rem; line-height: 1.1; font-weight: 900; color: #123e63; margin-top: .28rem;}
.metric-note {font-size: .76rem; color: #6c7e8e; margin-top: .18rem;}

.prediction-card {
  background: linear-gradient(145deg, #eef8ff, #ffffff);
  border: 2px solid #2e6f95;
  border-radius: 20px;
  padding: 1rem;
  text-align: center;
}
.pred-label {font-size: 1.62rem; font-weight: 900; color: #0d3557; margin: .18rem 0;}
.pred-confidence {font-size: 2.25rem; font-weight: 950; color: #14749d; line-height: 1.05;}
.pred-caption {font-size: .84rem; font-weight: 800; color: #557286; text-transform: uppercase; letter-spacing: .06em;}

.note {
  background: #eef6fb; border-left: 5px solid #28789f; color: #173c57;
  border-radius: 10px; padding: .72rem .85rem; font-size: .92rem; line-height: 1.45;
}
.warning-note {
  background: #fff7e6; border-left: 5px solid #d48a1f; color: #654313;
  border-radius: 10px; padding: .72rem .85rem; font-size: .92rem; line-height: 1.45;
}
.setup-card {
  background: #fff8e8; border: 1px solid #f1d28c; border-radius: 16px;
  padding: 1rem; color: #654313;
}

.result-table {width: 100%; border-collapse: separate; border-spacing: 0; font-size: .96rem; background: white;}
.result-table th {background: #123e63; color: white; padding: .66rem .62rem; text-align: center; font-weight: 900;}
.result-table td {padding: .62rem; border-bottom: 1px solid #e2eaf2; text-align: center; font-weight: 650;}
.result-table td:first-child, .result-table th:first-child {text-align: left;}
.result-table tr.proposed td {background: #e9f7ff; font-weight: 900; color: #0b4c72;}
.result-table tr:nth-child(even):not(.proposed) td {background: #f8fbfd;}

.prob-row {margin: .32rem 0 .56rem;}
.prob-head {display:flex; justify-content:space-between; font-weight:850; color:#264a66; font-size:.91rem;}
.prob-track {height: 13px; background:#e4edf4; border-radius:999px; overflow:hidden; margin-top:.25rem;}
.prob-fill {height:100%; background: linear-gradient(90deg, #2a6f97, #36a5c7); border-radius:999px;}

.small-muted {font-size: .82rem; color:#657889;}
.big-callout {font-size:1.08rem; line-height:1.55; font-weight:650; color:#23445d;}

[data-testid="stImage"] img {border-radius: 14px; object-fit: contain;}
[data-testid="stFileUploader"] {background:white; border-radius:16px; padding:.25rem .5rem;}
button[kind="primary"] {font-weight: 900;}

@media (max-width: 900px) {
  .hero-title {font-size:1.65rem;}
  .pred-label {font-size:1.35rem;}
  .pred-confidence {font-size:1.85rem;}
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Study results grounded in the supplied final notebooks
# -----------------------------------------------------------------------------
PROPOSED_METRICS = {
    "Accuracy": 0.956288,
    "Precision": 0.956351,
    "Recall": 0.956137,
    "F1 Score": 0.956220,
}

BASELINE_RESULTS = [
    {"Model": "VGG16", "Accuracy": 0.932515, "Precision": 0.932534, "Recall": 0.932599, "F1 Score": 0.932474},
    {"Model": "ResNet50", "Accuracy": 0.898006, "Precision": 0.897995, "Recall": 0.897946, "F1 Score": 0.897921},
    {"Model": "ConvNeXt Small", "Accuracy": 0.893405, "Precision": 0.893841, "Recall": 0.893533, "F1 Score": 0.893381},
    {"Model": "DenseNet201", "Accuracy": 0.870399, "Precision": 0.873902, "Recall": 0.870748, "F1 Score": 0.870533},
]

SELF_SEMI_RESULTS = [
    {"Model": "BYOL-VGG16", "Method": "Self-supervised", "Accuracy": 0.956288, "Precision": 0.956351, "Recall": 0.956137, "F1 Score": 0.956220},
    {"Model": "BYOL-ResNet50", "Method": "Self-supervised", "Accuracy": 0.930215, "Precision": 0.930590, "Recall": 0.930043, "F1 Score": 0.930183},
    {"Model": "BYOL-ConvNeXt Small", "Method": "Self-supervised", "Accuracy": 0.882669, "Precision": 0.882707, "Recall": 0.882914, "F1 Score": 0.882750},
    {"Model": "DINO-VGG16", "Method": "Self-supervised", "Accuracy": 0.899540, "Precision": 0.902721, "Recall": 0.898862, "F1 Score": 0.899346},
    {"Model": "DINO-ConvNeXt Small", "Method": "Self-supervised", "Accuracy": 0.890337, "Precision": 0.890747, "Recall": 0.889991, "F1 Score": 0.890139},
    {"Model": "DINO-ResNet50", "Method": "Self-supervised", "Accuracy": 0.836656, "Precision": 0.840113, "Recall": 0.837193, "F1 Score": 0.836890},
    {"Model": "STAC-VGG16", "Method": "Semi-supervised", "Accuracy": 0.832055, "Precision": 0.832962, "Recall": 0.831672, "F1 Score": 0.831807},
    {"Model": "STAC-ConvNeXt Small", "Method": "Semi-supervised", "Accuracy": 0.793712, "Precision": 0.794326, "Recall": 0.793836, "F1 Score": 0.793489},
    {"Model": "STAC-ResNet50", "Method": "Semi-supervised", "Accuracy": 0.784509, "Precision": 0.786106, "Recall": 0.784407, "F1 Score": 0.784629},
]

FALLBACK_CLASS_NAMES = ["Herniated Disc", "No Stenosis", "Thecal Sac"]
CLASS_NOTES = {
    "Herniated Disc": "Dataset class corresponding to the herniated-disc category used during training.",
    "No Stenosis": "Dataset class representing the no-stenosis category used during training.",
    "Thecal Sac": "Dataset class corresponding to the thecal-sac category used during training.",
}


# -----------------------------------------------------------------------------
# Exact proposed-model architecture and preprocessing from the BYOL notebook
# -----------------------------------------------------------------------------
class MRIPreprocess:
    def __init__(
        self,
        threshold: int = 10,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        padding: float = 0.04,
    ):
        self.threshold = threshold
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.padding = padding

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)

        foreground_mask = gray > self.threshold
        if foreground_mask.any():
            rows, columns = np.where(foreground_mask)
            top, bottom = rows.min(), rows.max() + 1
            left, right = columns.min(), columns.max() + 1

            vertical_padding = max(2, int((bottom - top) * self.padding))
            horizontal_padding = max(2, int((right - left) * self.padding))

            top = max(0, top - vertical_padding)
            bottom = min(gray.shape[0], bottom + vertical_padding)
            left = max(0, left - horizontal_padding)
            right = min(gray.shape[1], right + horizontal_padding)
            gray = gray[top:bottom, left:right]

        foreground = gray[gray > self.threshold]
        if foreground.size > 10:
            lower, upper = np.percentile(
                foreground,
                [self.lower_percentile, self.upper_percentile],
            )
            if upper > lower:
                gray = np.clip(
                    (gray.astype(np.float32) - lower) * 255.0 / (upper - lower),
                    0,
                    255,
                ).astype(np.uint8)

        height, width = gray.shape
        side = max(height, width)
        canvas = np.zeros((side, side), dtype=np.uint8)
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[top : top + height, left : left + width] = gray
        return Image.fromarray(canvas, mode="L").convert("RGB")


class TransferClassifier(nn.Module):
    def __init__(self, timm_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_count = int(
            self.backbone.head_hidden_size
            if hasattr(self.backbone, "head_hidden_size")
            else self.backbone.num_features
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.40),
            nn.Linear(feature_count, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.classifier(features)


def _torch_load(source, map_location):
    try:
        return torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(source, map_location=map_location)


def find_checkpoint() -> Path | None:
    if EXPECTED_CHECKPOINT.exists():
        return EXPECTED_CHECKPOINT

    candidates = sorted(MODEL_DIR.glob("*vgg16*byol*finetuned*best*.pth"))
    if not candidates:
        candidates = sorted(MODEL_DIR.glob("*.pth"))
    return candidates[0] if candidates else None


@st.cache_resource(show_spinner=False)
def load_proposed_model(checkpoint_path: str):
    payload = _torch_load(checkpoint_path, map_location=DEVICE)

    required = ["model_state_dict", "timm_name", "class_names", "mean", "std"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Checkpoint is missing required keys: {missing}")

    class_names = list(payload.get("class_names", FALLBACK_CLASS_NAMES))
    timm_name = str(payload.get("timm_name", "vgg16.tv_in1k"))
    input_size = int(payload.get("input_size", 224))
    mean = tuple(float(value) for value in payload["mean"])
    std = tuple(float(value) for value in payload["std"])

    if len(class_names) != 3:
        raise RuntimeError(
            f"Expected three classes, but checkpoint contains {len(class_names)} classes."
        )

    model = TransferClassifier(timm_name, len(class_names)).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            MRIPreprocess(),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    display_transform = transforms.Compose(
        [
            MRIPreprocess(),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )

    meta = {
        "class_names": class_names,
        "timm_name": timm_name,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "best_epoch": payload.get("best_epoch"),
        "best_val_f1": payload.get("best_val_f1"),
    }
    return model, transform, display_transform, meta


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    # VGG16-specific target: the last convolution before the final ReLU/pooling.
    # Using the final convolutional feature map gives the standard Grad-CAM signal.
    if hasattr(model.backbone, "features"):
        for module in reversed(list(model.backbone.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module

    last_conv = None
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No convolutional layer was found for Grad-CAM.")
    return last_conv


def gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    display_image: Image.Image,
    class_index: int,
) -> Image.Image:
    """Generate a conventional high-contrast Grad-CAM overlay."""
    import matplotlib.cm as cm

    target_layer = find_last_conv_layer(model)
    storage: Dict[str, torch.Tensor] = {}

    # IMPORTANT: VGG16's `features` block applies ReLU(inplace=True) directly
    # after every convolution. If we grab the gradient with
    # `register_full_backward_hook` on the Conv2d module itself, PyTorch raises
    # "Output 0 ... is a view and is being modified inplace" as soon as that
    # in-place ReLU runs on the very tensor the hook is watching - the
    # try/except around this function was silently swallowing that error and
    # falling back to the plain (non-heatmap) image, which is why Grad-CAM
    # never appeared. Registering the gradient hook directly on the output
    # *tensor* (captured before the in-place ReLU mutates it) avoids this
    # entirely and is the standard fix used by Grad-CAM implementations.
    def forward_hook(_module, _inputs, output):
        # Clone now, before the following in-place ReLU overwrites this
        # tensor's storage, so we keep the true pre-ReLU conv activations.
        storage["activation"] = output.detach().clone()

        def _capture_gradient(grad):
            storage["gradient"] = grad.detach().clone()

        if output.requires_grad:
            output.register_hook(_capture_gradient)

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            logits = model(x)
            score = logits[:, class_index].sum()
            score.backward()

        activation = storage.get("activation")
        gradient = storage.get("gradient")

        if activation is None or gradient is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        activation = activation.detach().float()
        gradient = gradient.detach().float()

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)

        if float(cam.max()) <= 0.0:
            # Some confident predictions can yield an all-zero map after ReLU.
            # Absolute gradient-weighted activation is a safe visualization fallback
            # while preserving the same target layer and predicted class.
            cam = torch.abs(
                torch.sum(weights * activation, dim=1, keepdim=True)
            )

        cam = F.interpolate(
            cam,
            size=(display_image.height, display_image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.cpu().numpy()
        if not np.isfinite(cam).all():
            raise RuntimeError("Grad-CAM produced non-finite values.")

        cam -= cam.min()
        cam_max = float(cam.max())
        if cam_max <= 1e-12:
            raise RuntimeError("Grad-CAM response is effectively empty.")
        cam /= cam_max

        # Suppress diffuse low activations and enhance the focal region, matching
        # the conventional red/yellow hotspot presentation used in papers.
        positive = cam[cam > 0]
        threshold = float(np.percentile(positive, 55)) if positive.size else 0.0
        cam = np.clip((cam - threshold) / max(1.0 - threshold, 1e-8), 0.0, 1.0)
        cam = np.power(cam, 0.65)

        # VGG16's last conv feature map is only 7x7 for a 224x224 input, so the
        # CAM is bilinearly upsampled ~32x to display size. That coarseness can
        # let a "hot" cell near the anatomy's edge bleed into the black padding
        # introduced by the square-pad preprocessing step. We know which pixels
        # are real anatomy (the same intensity threshold used during
        # preprocessing), so we dampen - but deliberately do NOT zero out -
        # the heatmap over background before blending. Damping (rather than a
        # hard cutoff applied before normalization) avoids a degenerate case
        # where the model's most class-discriminative signal happens to fall
        # right at the body's edge: a hard zero-out there can wipe out the
        # entire heatmap ("Grad-CAM response is effectively empty"), while
        # damping only fades a background bleed without ever fully erasing it.
        gray_display = np.asarray(display_image.convert("L"), dtype=np.float32)
        body_mask_img = Image.fromarray((gray_display > 10.0).astype(np.uint8) * 255)
        body_mask_img = body_mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        body_mask = np.asarray(body_mask_img, dtype=np.float32) / 255.0
        body_mask = np.clip(body_mask, 0.2, 1.0)  # dampen, never fully zero

        base = np.asarray(display_image.convert("RGB"), dtype=np.float32) / 255.0
        heat = cm.get_cmap("jet")(cam)[..., :3].astype(np.float32)

        # Conventional CAM overlay: visible blue tint with a strong yellow/red
        # hotspot over tissue, like the reference paper figure. The blend
        # weight is dampened by body_mask so background pixels only get a
        # faint tint instead of a strong false-color wash, while still never
        # fully hiding a hotspot that happens to sit near the body's edge.
        alpha = (0.52 * body_mask)[..., None]
        overlay = np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0)

        return Image.fromarray((overlay * 255.0).astype(np.uint8), mode="RGB")
    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)


def analyze_input_quality(image: Image.Image) -> List[str]:
    warnings: List[str] = []
    if image.width < 128 or image.height < 128:
        warnings.append("The uploaded image is low resolution; prediction may be less reliable.")

    sample = np.asarray(image.convert("RGB").resize((192, 192)), dtype=np.float32)
    channel_difference = (
        np.abs(sample[..., 0] - sample[..., 1]).mean()
        + np.abs(sample[..., 1] - sample[..., 2]).mean()
        + np.abs(sample[..., 0] - sample[..., 2]).mean()
    ) / 3.0
    if channel_difference > 18.0:
        warnings.append(
            "The image is strongly colored and does not resemble the grayscale MRI appearance used for training."
        )

    gray = np.asarray(image.convert("L").resize((192, 192)), dtype=np.float32)
    if gray.std() < 8.0:
        warnings.append("The image has very low intensity variation; verify the MRI image before interpretation.")
    return warnings


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def result_table_html(rows: List[dict], include_method: bool = False) -> str:
    headers = ["Model"] + (["Learning Type"] if include_method else []) + [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    parts = ["<table class='result-table'><thead><tr>"]
    parts.extend(f"<th>{header}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")

    for row in rows:
        proposed = row["Model"] == "BYOL-VGG16"
        parts.append("<tr class='proposed'>" if proposed else "<tr>")
        parts.append(f"<td>{row['Model']}{' ★' if proposed else ''}</td>")
        if include_method:
            parts.append(f"<td>{row['Method']}</td>")
        for key in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            parts.append(f"<td>{percent(row[key])}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


def probability_html(class_names: List[str], probabilities: np.ndarray) -> str:
    order = np.argsort(probabilities)[::-1]
    blocks = []
    for index in order:
        probability = float(probabilities[index])
        width = max(1.2, 100.0 * probability)
        blocks.append(
            f"""
            <div class='prob-row'>
              <div class='prob-head'><span>{class_names[index]}</span><span>{100*probability:.2f}%</span></div>
              <div class='prob-track'><div class='prob-fill' style='width:{width:.2f}%'></div></div>
            </div>
            """
        )
    return "".join(blocks)


def make_paper_figure(
    original: Image.Image,
    model_input: Image.Image,
    cam_image: Image.Image,
    class_names: List[str],
    probabilities: np.ndarray,
    predicted_class: str,
) -> bytes:
    import matplotlib.pyplot as plt

    order = np.argsort(probabilities)[::-1]
    fig = plt.figure(figsize=(14.5, 4.8), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.28)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(model_input.convert("L"), cmap="gray")
    ax1.set_title("Model Input", fontsize=16, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(cam_image)
    ax2.set_title("Grad-CAM Heatmap", fontsize=16, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(grid[0, 2])
    labels = [class_names[i] for i in order]
    values = [100 * float(probabilities[i]) for i in order]
    ax3.barh(labels[::-1], values[::-1])
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Prediction probability (%)", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="both", labelsize=11)
    ax3.grid(axis="x", alpha=0.22)
    ax3.set_title(f"Prediction: {predicted_class}", fontsize=16, fontweight="bold")

    fig.suptitle(
        "SpineSight MRI — Proposed BYOL-VGG16 Inference",
        fontsize=19,
        fontweight="bold",
        y=1.02,
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Header and study metrics
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class='hero'>
  <div class='hero-title'>🩻 SpineSight MRI</div>
  <div class='hero-sub'>Lumbar spinal MRI classification and visual explanation using the proposed BYOL-VGG16 model</div>
  <div class='hero-badge'>Research Prototype · 3-Class MRI Classification · 224 × 224</div>
</div>
""",
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
for column, (name, value) in zip(metric_columns, PROPOSED_METRICS.items()):
    with column:
        st.markdown(
            f"""
            <div class='metric-card'>
              <div class='metric-name'>{name}</div>
              <div class='metric-value'>{percent(value)}</div>
              <div class='metric-note'>BYOL-VGG16 test result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
tab_analysis, tab_results, tab_model = st.tabs(
    ["MRI Analysis", "Study Results", "Model Card"]
)

checkpoint_path = find_checkpoint()
model_bundle = None
model_error = None
if checkpoint_path is not None:
    try:
        with st.spinner("Loading proposed BYOL-VGG16 checkpoint..."):
            model_bundle = load_proposed_model(str(checkpoint_path))
    except Exception as exc:
        model_error = str(exc)


# -----------------------------------------------------------------------------
# MRI analysis tab
# -----------------------------------------------------------------------------
with tab_analysis:
    top_left, top_right = st.columns([1.25, 0.75], gap="large")
    with top_left:
        st.markdown("<div class='section-title'>Analyze a lumbar MRI image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload JPG, JPEG, or PNG",
            type=["jpg", "jpeg", "png"],
            help="Use an image with appearance similar to the MRI images used in the study.",
        )
    with top_right:
        st.markdown(
            """
            <div class='note'>
            <b>Proposed model:</b> BYOL-VGG16<br>
            <b>Classes:</b> Herniated Disc · No Stenosis · Thecal Sac<br>
            <b>Purpose:</b> research demonstration, not clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if checkpoint_path is None:
        st.markdown(
            """
            <div class='setup-card'>
            <b>Model checkpoint not found.</b><br>
            Put <code>vgg16_byol_finetuned_best.pth</code> inside the app's <code>models/</code> folder,
            then restart Streamlit. The Study Results and Model Card tabs work without the checkpoint.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif model_error:
        st.error(f"Checkpoint could not be loaded: {model_error}")
    elif uploaded_file is None:
        st.markdown(
            """
            <div class='card'>
              <div class='section-label'>Ready</div>
              <div class='big-callout'>Upload one MRI image to display the predicted class, softmax confidence distribution, exact preprocessed model input, and Grad-CAM localization.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"The uploaded image could not be read: {exc}")
            st.stop()

        model, transform, display_transform, meta = model_bundle
        quality_warnings = analyze_input_quality(original_image)
        for warning in quality_warnings:
            st.markdown(f"<div class='warning-note'>⚠️ {warning}</div>", unsafe_allow_html=True)

        model_input_image = display_transform(original_image)
        input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

        # torch.no_grad() (rather than torch.inference_mode()) is used here
        # because the same cached model/input are reused for gradient-based
        # Grad-CAM right below; no_grad() keeps tensors as ordinary (non
        # "inference") tensors so there's no risk of interference with the
        # autograd pass that follows.
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = meta["class_names"][predicted_index]
        confidence = float(probabilities[predicted_index])

        # Generate Grad-CAM before laying out the main result row so the
        # paper screenshot can show Model Input → Grad-CAM → Prediction.
        try:
            with st.spinner("Generating Grad-CAM..."):
                cam_image = gradcam_overlay(
                    model,
                    input_tensor,
                    model_input_image,
                    predicted_index,
                )
            cam_error = None
        except Exception as exc:
            cam_image = model_input_image
            cam_error = str(exc)

        input_col, cam_col, prediction_col = st.columns(
            [1.0, 1.0, 0.92],
            gap="large",
        )

        with input_col:
            st.markdown(
                "<div class='section-label'>Model Input</div>",
                unsafe_allow_html=True,
            )
            st.image(model_input_image, use_container_width=True)
            st.caption(
                "Foreground crop · intensity normalization · "
                "square padding · 224×224 resize"
            )

        with cam_col:
            st.markdown(
                "<div class='section-label'>Grad-CAM Heatmap</div>",
                unsafe_allow_html=True,
            )
            st.image(cam_image, use_container_width=True)
            if cam_error:
                st.warning(f"Grad-CAM could not be generated: {cam_error}")
            else:
                st.caption(
                    "Red/yellow = stronger contribution · blue = weaker contribution"
                )

        with prediction_col:
            st.markdown(
                "<div class='section-label'>Prediction</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class='prediction-card'>
                  <div class='pred-caption'>Predicted class</div>
                  <div class='pred-label'>{predicted_class}</div>
                  <div class='pred-caption' style='margin-top:.5rem'>Softmax confidence</div>
                  <div class='pred-confidence'>{100*confidence:.2f}%</div>
                </div>
                <div class='note' style='margin-top:.65rem'>
                {CLASS_NOTES.get(predicted_class, 'Dataset class prediction.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='card'><div class='section-label'>Class Probabilities</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            probability_html(meta["class_names"], probabilities),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            paper_png = make_paper_figure(
                original_image,
                model_input_image,
                cam_image,
                meta["class_names"],
                probabilities,
                predicted_class,
            )
            st.download_button(
                "Download paper-ready analysis panel (300 DPI PNG)",
                data=paper_png,
                file_name="spinesight_mri_analysis.png",
                mime="image/png",
                use_container_width=True,
                type="primary",
            )
        except Exception as exc:
            st.caption(f"Paper-figure export unavailable: {exc}")

        st.markdown(
            "<div class='warning-note'><b>Research-use note:</b> Softmax confidence is not a calibrated clinical probability. The application reproduces the paper model's image-classification pipeline and should not be used as a standalone diagnostic system.</div>",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Results tab
# -----------------------------------------------------------------------------
with tab_results:
    st.markdown("<div class='section-title'>Top supervised baseline models</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>The supplied final notebooks contain eight supervised backbones in total. The four highest supervised F1 scores are shown below.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(result_table_html(BASELINE_RESULTS), unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='section-title'>Self- and semi-supervised results</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note'>BYOL-VGG16 achieves the highest F1 score among the supplied final results and is therefore presented in this application as the proposed model.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(result_table_html(SELF_SEMI_RESULTS, include_method=True), unsafe_allow_html=True)

    st.write("")
    comparison = pd.DataFrame(
        [
            {"Model": row["Model"], "F1 Score": row["F1 Score"]}
            for row in (BASELINE_RESULTS + SELF_SEMI_RESULTS)
        ]
    ).sort_values("F1 Score", ascending=False)
    st.markdown("<div class='section-label'>Overall F1 Ranking</div>", unsafe_allow_html=True)
    st.bar_chart(comparison.set_index("Model")["F1 Score"], height=420)


# -----------------------------------------------------------------------------
# Model card tab
# -----------------------------------------------------------------------------
with tab_model:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(
            """
            <div class='card'>
              <div class='section-label'>Proposed Architecture</div>
              <div class='section-title'>BYOL-VGG16</div>
              <div class='big-callout'>
              The VGG16 backbone is initialized from ImageNet weights, adapted through BYOL self-supervised pretraining on the training split with labels discarded, and then fine-tuned for the three-class lumbar MRI task.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='card'>
              <div class='section-label'>Image Pipeline</div>
              <div class='big-callout'>
              Foreground-aware crop → robust 1st–99th percentile intensity normalization → square padding → RGB replication → 224×224 bicubic resize → pretrained-model normalization.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class='card'>
              <div class='section-label'>Experimental Protocol</div>
              <div class='big-callout'>
              <b>Outer split:</b> 70% train · 20% validation · 10% test<br>
              <b>Fine-tuning batch size:</b> 32<br>
              <b>Head training:</b> 5 epochs<br>
              <b>Full fine-tuning:</b> up to 25 epochs with early stopping<br>
              <b>Checkpoint selection:</b> validation macro-F1
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        status = "Available" if checkpoint_path is not None and not model_error else "Not loaded"
        path_text = checkpoint_path.name if checkpoint_path else "models/vgg16_byol_finetuned_best.pth"
        st.markdown(
            f"""
            <div class='card'>
              <div class='section-label'>Deployment Status</div>
              <div class='big-callout'>
              <b>Checkpoint:</b> {status}<br>
              <b>Expected file:</b> {path_text}<br>
              <b>Runtime device:</b> {DEVICE.type.upper()}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='warning-note'><b>Scope:</b> This application reflects the labels, model architecture, preprocessing, and reported metrics contained in the supplied notebooks. It is intended for research presentation and reproducible inference, not for clinical decision-making.</div>",
        unsafe_allow_html=True,
    )
