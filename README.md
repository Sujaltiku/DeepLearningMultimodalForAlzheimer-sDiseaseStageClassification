#Multimodal Alzheimer's Disease Classification

A deep learning framework for early detection and classification of Alzheimer's Disease using multimodal neuroimaging — fusing MRI and PET scan data with an interactive UI for clinical inference.

---

## 📁 Project Structure

```
alzheimers-classification/
│
├── MRI Preprocessing/        # MRI scan preprocessing pipeline
│   ├── skull_stripping.py
│   ├── registration.py
│   ├── normalization.py
│   └── mri_pipeline.ipynb
│
├── PET Preprocessing/        # PET scan preprocessing pipeline
│   ├── pet_pipeline.py
│   ├── standardization.py
│   └── pet_visualizer.ipynb
│
├── UI/                       # Frontend interface for inference
│   ├── app.py (or index.js)
│   └── requirements.txt
│
├── model.ipynb               # Main model training & evaluation notebook
├── MNI152_T1_1mm.nii.gz     # MNI152 standard brain template (11 MB)
├── requirements.txt          # Python dependencies
└── README.md
```

---

##  Overview

This project implements a **multimodal deep learning pipeline** for classifying Alzheimer's Disease stages using:

- **Structural MRI** — volumetric brain atrophy analysis
- **PET Scans** — metabolic activity mapping
- **MNI152 Atlas** — standard space registration for spatial normalization

### Classification Classes
| Label | Description |
|-------|-------------|
| CN | Cognitively Normal |
| MCI | Mild Cognitive Impairment |
| AD | Alzheimer's Disease |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.8+
- PyTorch / TensorFlow
- NiBabel
- ANTsPy (for MRI registration)
- SimpleITK
- Nilearn

### Dataset

This project uses data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

>  Data is **not included** in this repository. You must apply for access at [adni.loni.usc.edu](https://adni.loni.usc.edu/).

Place your data as follows:
```
data/
├── raw/
│   ├── MRI/
│   └── PET/
└── processed/
```

---

## 🧪 Pipeline

### 1. MRI Preprocessing
```bash
cd "MRI Preprocessing"
# Run  registration to MNI152 space, skull stripping,and normalization
```

### 2. PET Preprocessing
```bash
cd "PET Preprocessing"
# Run standardization and alignment
```

### 3. Model Training
Open and run `model.ipynb` in Jupyter:
```bash
jupyter notebook model.ipynb
```

### 4. Launch UI
```bash
cd UI
```

---

##  Model Architecture

- **Multimodal Fusion** of MRI + PET features
- 3D CNN backbone for volumetric feature extraction
- Late fusion / attention-based feature aggregation
- Classification head for CN / MCI / AD prediction

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| AUC-ROC | TBD |
| F1-Score | TBD |

> Update this table after training with your dataset.

---

## 🗂️ MNI152 Template

The `MNI152_T1_1mm.nii.gz` file is the standard MNI152 brain template used for spatial normalization of all input scans to a common reference space.

---

## 📌 TODO

- [ ] Add model weights / checkpoint download link
- [ ] Add sample inference notebook
- [ ] Add data preprocessing validation scripts
- [ ] Docker support

---

## 📄 License

This project is for academic/research purposes. Dataset usage is subject to [ADNI's data sharing agreement](https://adni.loni.usc.edu/data-samples/access-data/).

---

## 🙏 Acknowledgements

- [ADNI](https://adni.loni.usc.edu/) for neuroimaging data
- [MNI152 Atlas](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009) for brain template
- [NiBabel](https://nipy.org/nibabel/) & [ANTsPy](https://github.com/ANTsX/ANTsPy) for neuroimaging utilities
