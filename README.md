# LumaRep

### An Interpretable Lumbar Spinal MRI Classification Framework via Self-Supervised Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Self-Supervised Learning](https://img.shields.io/badge/Learning-BYOL%20%7C%20DINO-6f42c1)
![Streamlit](https://img.shields.io/badge/Streamlit-Research%20App-ff4b4b)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20Occlusion%20Sensitivity-2ea44f)
![Status](https://img.shields.io/badge/Status-Accepted%20at%20CSNT%202027-success)

**LumaRep** is an interpretable deep-learning framework for **three-class lumbar spinal MRI classification**. The study compares supervised CNN baselines with **self-supervised learning (BYOL and DINO)** and **semi-supervised learning (STAC)**, then selects **BYOL-VGG16** as the proposed model. Explainability is supported through **Grad-CAM** and **occlusion sensitivity**, and a lightweight **Streamlit research prototype** demonstrates image-level inference and visual explanation.

> **Status:** Accepted at **CSNT 2027**. Official bibliographic information and DOI will be added when available.

---

## Highlights

- **13,036** processed lumbar MRI images after duplicate removal
- **3 classes:** Herniated Disc, No Stenosis, and Thecal Sac
- Supervised baselines: **VGG16, ResNet50, ConvNeXt Small, DenseNet201**
- Self-supervised methods: **BYOL** and **DINO**
- Semi-supervised method: **STAC**
- Proposed framework: **BYOL-VGG16**
- Best reported accuracy: **95.63%**
- Explainability: **Grad-CAM** and **occlusion sensitivity**
- Interactive research prototype built with **Streamlit**

---

## Method Overview

The experimental workflow is organized as follows:

1. Prepare and clean the publicly available lumbar spine MRI dataset.
2. Remove duplicate images and construct a stratified **70:20:10** train/validation/test split.
3. Establish supervised CNN baselines.
4. Evaluate **BYOL** and **DINO** self-supervised representation learning.
5. Evaluate **STAC** as a semi-supervised approach.
6. Compare the learning strategies across selected CNN backbones.
7. Select **BYOL-VGG16** as the final LumaRep model.
8. Analyze predictions using **Grad-CAM** and **occlusion sensitivity**.
9. Demonstrate inference through a Streamlit research application.

---

## Performance

### Baseline models

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| VGG16 | 93.25% | 93.25% | 93.26% | 93.25% |
| ResNet50 | 89.80% | 89.80% | 89.79% | 89.79% |
| ConvNeXt Small | 89.34% | 89.38% | 89.35% | 89.34% |
| DenseNet201 | 87.04% | 87.39% | 87.07% | 87.05% |

### Self- and semi-supervised models

| Backbone | Method | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|
| **VGG16** | **BYOL** | **95.63%** | **95.64%** | **95.61%** | **95.62%** |
| ResNet50 | BYOL | 93.02% | 93.06% | 93.00% | 93.02% |
| ConvNeXt Small | BYOL | 88.27% | 88.27% | 88.29% | 88.28% |
| VGG16 | DINO | 89.95% | 90.27% | 89.89% | 89.93% |
| ResNet50 | DINO | 83.67% | 84.01% | 83.72% | 83.69% |
| ConvNeXt Small | DINO | 89.03% | 89.07% | 89.00% | 89.01% |
| VGG16 | STAC | 83.21% | 83.30% | 83.17% | 83.18% |
| ResNet50 | STAC | 78.45% | 78.61% | 78.44% | 78.46% |
| ConvNeXt Small | STAC | 79.37% | 79.43% | 79.38% | 79.35% |

The final **LumaRep (BYOL-VGG16)** model reports **95.63% accuracy, 95.64% precision, 95.61% recall, and 95.62% F1 score**.

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
│       └── README.md
└── paper/
    └── README.md
```

The `paper/` directory contains publication and citation information. The publisher-formatted conference paper is not redistributed in this repository.

---

## Notebooks

### `01_supervised_baseline_screening.ipynb`
Supervised baseline experiments used to compare pretrained CNN architectures for three-class lumbar spinal MRI classification.

### `02_convnext_small_baseline.ipynb`
Additional ConvNeXt Small baseline experiments.

### `03_efficientnetb3_exploratory.ipynb`
Exploratory EfficientNetB3 experiments retained for research transparency.

### `04_byol_self_supervised.ipynb`
BYOL-based self-supervised pretraining and downstream fine-tuning. The **BYOL-VGG16** configuration produced the best overall result and was selected as LumaRep.

### `05_dino_self_supervised.ipynb`
DINO teacher-student self-distillation experiments across selected CNN backbones.

### `06_stac_semi_supervised.ipynb`
STAC-based semi-supervised experiments using labeled and pseudo-labeled training samples.

---

## Streamlit Research Application

The repository includes a Streamlit prototype for interactive inference with the proposed **BYOL-VGG16** model.

The application:

- accepts a lumbar MRI image,
- predicts one of the three study classes,
- reports softmax confidence,
- displays the exact model input,
- generates a **Grad-CAM** visualization,
- provides experiment-summary and model-card views.

### Required checkpoint

The trained checkpoint is intentionally not tracked in the normal Git repository.

Expected filename:

```text
vgg16_byol_finetuned_best.pth
```

Place it inside:

```text
app/models/vgg16_byol_finetuned_best.pth
```

### Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

> **Research-use note:** The application is intended for research and demonstration purposes and is **not a standalone clinical diagnostic system**. Softmax confidence should not be interpreted as a calibrated clinical probability.

---

## Dataset

The study uses the publicly available **Lumbar Spine MRI Dataset** from Mendeley Data.

- Original images: **13,686**
- Duplicate images removed: **650**
- Final processed images: **13,036**
- Split: **70% training / 20% validation / 10% testing**
- Classes:
  - **Herniated Disc:** 4,215 images
  - **No Stenosis:** 4,400 images
  - **Thecal Sac:** 4,421 images
- Dataset DOI: `10.17632/k57fr854j2.2`

The dataset itself is **not redistributed** in this repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shawna175/LumaRep.git
cd LumaRep
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project primarily uses:

- PyTorch
- torchvision
- timm
- NumPy
- pandas
- Pillow
- Matplotlib
- Streamlit

---

## Explainability

LumaRep uses **Grad-CAM** and **occlusion sensitivity** to investigate image regions that influence model predictions.

These methods provide visual insight into model attention and prediction sensitivity. They are intended to support transparency and research analysis rather than serve as independent evidence of disease.

---

## Limitations

The study was evaluated on a **single curated lumbar MRI dataset** and did not include an independent external clinical cohort. This may limit assessment of generalizability across institutions, imaging systems, and real-world clinical settings.

Future work includes:

- multi-center external validation,
- larger and more diverse datasets,
- additional lumbar conditions,
- patient-level evaluation,
- further investigation of self-supervised and hybrid representation-learning methods.

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

Temporary citation:

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

No open-source license is currently attached to this repository. Please contact the authors before reusing the code or model weights beyond academic review and reference.
