# LumaRep

### Interpretable Lumbar Spinal MRI Classification via Self-Supervised Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![SSL](https://img.shields.io/badge/Self--Supervised-BYOL%20%7C%20DINO-6f42c1)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20Occlusion%20Sensitivity-2ea44f)
![Streamlit](https://img.shields.io/badge/Streamlit-Research%20Prototype-ff4b4b)
![Status](https://img.shields.io/badge/Status-Accepted%20at%20CSNT%202027-success)

**LumaRep** is an interpretable deep-learning framework for **three-class lumbar spinal MRI classification**. The study compares supervised CNN baselines with **BYOL** and **DINO** self-supervised learning and **STAC** semi-supervised learning. The best-performing configuration, **BYOL-VGG16**, is selected as the proposed LumaRep model and is analyzed using **Grad-CAM** and **occlusion sensitivity**. A lightweight **Streamlit research prototype** demonstrates image-level inference and visual explanation.

> **Status:** Accepted at **CSNT 2027**. Official bibliographic information and DOI will be added when available.

---

## Key Contributions

- Curated **13,036 MRI images** after duplicate removal
- Three study classes: **Herniated Disc, No Stenosis, Thecal Sac**
- Supervised baseline comparison using **VGG16, ResNet50, ConvNeXt Small, and DenseNet201**
- Self-supervised representation learning using **BYOL** and **DINO**
- Semi-supervised comparison using **STAC**
- Final proposed model: **BYOL-VGG16**
- Explainability using **Grad-CAM** and **occlusion sensitivity**
- Streamlit-based research interface for interactive inference

---

## Results

| Configuration | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| VGG16 baseline | 93.25% | 93.25% | 93.26% | 93.25% |
| ResNet50 baseline | 89.80% | 89.80% | 89.79% | 89.79% |
| ConvNeXt Small baseline | 89.34% | 89.38% | 89.35% | 89.34% |
| DenseNet201 baseline | 87.04% | 87.39% | 87.07% | 87.05% |
| **BYOL-VGG16 (LumaRep)** | **95.63%** | **95.64%** | **95.61%** | **95.62%** |
| DINO-VGG16 | 89.95% | 90.27% | 89.89% | 89.93% |
| STAC-VGG16 | 83.21% | 83.30% | 83.17% | 83.18% |

**BYOL-VGG16** achieved the strongest overall performance and was selected as the final LumaRep framework.

---

## Method Overview

1. Prepare and clean the publicly available lumbar spine MRI dataset.
2. Remove duplicate images and create a stratified **70:20:10** train/validation/test split.
3. Establish supervised CNN baselines.
4. Evaluate **BYOL** and **DINO** self-supervised learning.
5. Evaluate **STAC** semi-supervised learning.
6. Compare the strongest learning configurations.
7. Select **BYOL-VGG16** as the proposed model.
8. Analyze predictions using **Grad-CAM** and **occlusion sensitivity**.
9. Demonstrate inference through a Streamlit research application.

---

## Repository Structure

```text
LumaRep/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_supervised_baseline_screening.ipynb
│   ├── 02_convnext_small_baseline.ipynb
│   ├── 03_efficientnetb3_exploratory.ipynb
│   ├── 04_byol_self_supervised.ipynb
│   ├── 05_dino_self_supervised.ipynb
│   └── 06_stac_semi_supervised.ipynb
├── app/
│   ├── app.py
│   ├── .streamlit/
│   │   └── config.toml
│   └── models/
│       └── PUT_MODEL_HERE.txt
└── paper/
    └── README.md
```

The `paper/` directory contains publication and citation information. The publisher-formatted conference paper is not redistributed in this repository.

---

## Notebooks

- **`01_supervised_baseline_screening.ipynb`** — supervised baseline screening and model comparison
- **`02_convnext_small_baseline.ipynb`** — ConvNeXt Small baseline experiments
- **`03_efficientnetb3_exploratory.ipynb`** — exploratory EfficientNetB3 experiments
- **`04_byol_self_supervised.ipynb`** — BYOL pretraining and downstream fine-tuning
- **`05_dino_self_supervised.ipynb`** — DINO self-distillation experiments
- **`06_stac_semi_supervised.ipynb`** — STAC-based semi-supervised experiments

---

## Streamlit Research Application

The repository includes a Streamlit prototype for interactive inference using the proposed **BYOL-VGG16** model.

The application supports:

- lumbar MRI image upload,
- three-class prediction,
- prediction confidence,
- model-input visualization,
- **Grad-CAM** explanation,
- baseline and self-/semi-supervised result summaries,
- model-card style experiment information.

### Required model checkpoint

The trained checkpoint is not tracked in the normal Git repository.

Expected filename:

```text
vgg16_byol_finetuned_best.pth
```

Place it at:

```text
app/models/vgg16_byol_finetuned_best.pth
```

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

> **Research-use note:** This application is intended for research and demonstration purposes and is not a standalone clinical diagnostic system.

---

## Dataset

The study uses the publicly available **Lumbar Spine MRI Dataset** from Mendeley Data.

- Original images: **13,686**
- Duplicate images removed: **650**
- Final processed images: **13,036**
- Split: **70% training / 20% validation / 10% testing**
- Herniated Disc: **4,215**
- No Stenosis: **4,400**
- Thecal Sac: **4,421**
- Dataset DOI: `10.17632/k57fr854j2.2`

The dataset itself is **not redistributed** in this repository.

---

## Installation

```bash
git clone https://github.com/shawna175/LumaRep.git
cd LumaRep
pip install -r requirements.txt
```

Core dependencies include **PyTorch, torchvision, timm, NumPy, pandas, Pillow, Matplotlib, and Streamlit**.

---

## Explainability

LumaRep uses **Grad-CAM** and **occlusion sensitivity** to investigate image regions that influence model predictions. These visualizations are intended to improve transparency and support research analysis rather than provide independent clinical evidence.

---

## Limitations

The study was evaluated on a **single curated lumbar MRI dataset** without an independent external clinical cohort. This limits assessment of generalizability across institutions, scanners, and real-world clinical settings.

Future work includes multi-center validation, larger and more diverse datasets, additional lumbar conditions, patient-level evaluation, and further investigation of self-supervised and hybrid representation-learning methods.

---

## Authors

- **Shawna Akter** — Department of CSE, East West University  
  [GitHub](https://github.com/shawna175) · [LinkedIn](https://www.linkedin.com/in/shawna-akter)
- **Mahfuz Uddin Ahmed** — Department of CSE, East West University
- **Moin Uddin Ahmed** — Department of CSE, East West University
- **Mustari Zaman** — Department of CSE, East West University
- **Md Mahfuzur Rahman** — Southern Arkansas University
- **Rafid Bin Taher** — Department of CSE, East West University

---

## Citation

Official conference citation and DOI will be added once the final bibliographic record becomes available.

```bibtex
@misc{akter2027lumarep,
  title  = {LumaRep: An Interpretable Lumbar Spinal MRI Classification via Self-Supervised Learning},
  author = {Akter, Shawna and Ahmed, Mahfuz Uddin and Ahmed, Moin Uddin and Zaman, Mustari and Rahman, Md Mahfuzur and Taher, Rafid Bin},
  note   = {Accepted at CSNT 2027},
  year   = {2027}
}
```

---

## License

No open-source license is currently attached to this repository.
