# Alzheimer's EEG Binary Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?logo=kaggle)](https://kaggle.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE-1.6.0-9B59B6)](https://mne.tools/)

---

## Clinical Motivation

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder with a long preclinical phase. Early detection using low‑cost, non‑invasive tools like **electroencephalography (EEG)** could revolutionise screening. This project classifies resting‑state EEG into **Normal** vs **Impaired** (Mild Cognitive Impairment + Moderate/Severe AD) using deep learning.

We evaluate three architectures: **EEGNet**, **EEGTransformer**, and **EEGGAT** (Graph Attention Network). The best model achieves **74% test accuracy** and **80% validation balanced accuracy**, demonstrating that EEG alone can capture disease‑related signatures.

---

## Dataset

- **Source:** OpenNeuro – **DS007526**  
  *Title: Resting‑state EEG in Alzheimer's disease, Frontotemporal dementia and healthy controls*  
  [https://openneuro.org/datasets/ds007526](https://openneuro.org/datasets/ds007526)
- **Subjects:** 88 (Alzheimer’s, Frontotemporal dementia, healthy controls)
- **Channels:** 19 (standard 10‑20 system: Fp1, Fp2, F3, F4, C3, C4, etc.)
- **Sampling rate:** 500 Hz (resampled to 250 Hz)
- **Recording:** Eyes‑closed resting‑state, ~5 minutes per subject
- **Labels:** Binary – **Normal** (MMSE ≥26) vs **Impaired** (MMSE <26, including MCI and moderate/severe AD)

> **Note:** The original dataset contains three groups; we merged MCI and moderate/severe into a single “Impaired” class to create a clinically relevant binary screening task.

---

## Repository Structure

alzheimers-eeg-classifier/
│
├── .github/workflows/ # CI (continuous integration) – optional
│ └── ci.yml
│
├── docs/ 
│ ├── reports/ 
│ │ ├── EEGNet_Report.md
│ │ ├── Transformer_Report.md
│ │ └── GAT_Report.md
│ └── clinical_background.md
│
├── experiments/ 
│ ├── config_eegnet.yaml 
│ ├── config_transformer.yaml
│ ├── config_gat.yaml
│ └── results/ 
│ ├── eegnet/
│ │ ├── best_model.pth
│ │ └── confusion_matrix.png
│ ├── transformer/
│ │ ├── best_model.pth
│ │ └── confusion_matrix.png
│ └── gat/
│ ├── best_model.pth
│ └── confusion_matrix.png
│
├── examples/ 
│ └── inference.py 
│
├── notebooks/ 
│ ├── 01_eegnet_training.ipynb
│ ├── 02_transformer_training.ipynb
│ └── 03_gat_training.ipynb
│
├── scripts/ 
│ └── train.py
│
├── src/ 
│ ├── init.py
│ ├── models.py 
│ ├── training.py 
│ └── utils.py 
│
├── tests/ 
│ └── test_preprocessing.py
│
├── .gitignore
├── LICENSE 
├── pyproject.toml 
├── requirements.txt 
└── README.md


---

## Models & Performance

We trained and compared three deep learning architectures on the same preprocessing pipeline and subject‑wise split.

### Preprocessing (identical for all models)
- Bandpass filter 0.5–45 Hz
- Notch filter 50 Hz
- Resample to 250 Hz
- 2‑second non‑overlapping epochs (500 timepoints)
- Z‑score normalisation per channel
- Subject‑wise stratified split (70% train, 15% val, 15% test)

### Results summary (test set)

| Model          | Test Accuracy | Balanced Accuracy (Val) | Precision (Impaired) | Recall (Impaired) | F1 (Impaired) |
|----------------|---------------|-------------------------|----------------------|-------------------|----------------|
| **EEGNet**     | 66.0%         | 72.8%                   | 0.88                 | 0.62              | 0.73           |
| **EEGTransformer** | **74.1%** | 75.5% (val accuracy)    | 0.79                 | 0.88              | **0.83**       |
| **EEGGAT**     | 65.8%         | 63.8%                   | 0.84                 | 0.66              | 0.74           |

> **Interpretation:** The Transformer model achieved the highest test accuracy (74%) and excellent recall for the Impaired class (88%), making it the most clinically useful for screening. EEGNet had higher precision for Impaired (88%) but lower recall. The GAT underperformed, likely due to limited graph connectivity information from only 19 channels.

### Detailed classification reports

<details>
<summary><b>EEGNet (final)</b></summary>
precision recall f1-score support
Normal 0.43 0.78 0.55 2553
Impaired 0.88 0.62 0.73 6899
accuracy 0.66 9452
macro avg 0.66 0.70 0.64 9452
weighted avg 0.76 0.66 0.68 9452
</details>
<details>
<summary><b>EEGTransformer</b></summary>
precision recall f1-score support
Normal 0.65 0.53 0.42 2553
Impaired 0.79 0.88 0.83 6899
accuracy 0.74 9452
macro avg 0.66 0.62 0.63 9452
weighted avg 0.72 0.74 0.72 9452
</details>
<details>
<summary><b>EEGGAT</b></summary>
precision recall f1-score support
Normal 0.41 0.65 0.51 2553
Impaired 0.84 0.66 0.74 6899
accuracy 0.66 9452
macro avg 0.63 0.65 0.62 9452
weighted avg 0.72 0.66 0.68 9452
</details>

---

## How to Reproduce

### 1. Clone the repository

git clone https://github.com/BurhanxGodhra/alzheimers-eeg-classifier.git
#cd alzheimers-eeg-classifier

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the dataset
Request access to OpenNeuro DS007526.

Place the extracted files in data/raw/ (create the folder).

### 4. Run preprocessing and training
Use the Kaggle notebooks in notebooks/ (upload to Kaggle) or run the Python scripts locally after adapting paths.

### 5. Run inference on a new EEG file
bash
python examples/inference.py --eeg_file path/to/subject_eeg.set --model experiments/results/transformer/best_model.pth

---

### Reports & Documentation
Detailed, formal reports describing the entire journey – from initial ternary classification to binary optimisation, architectural choices, hyperparameter tuning, and final results – are available in:

- EEGNet Report
- Transformer Report
- GAT Report

Also see the clinical background for the medical rationale.

---

### Key Takeaways
- Transformer outperformed EEGNet and GAT on this dataset, achieving 74% test accuracy.
- Focal loss and data augmentation (noise, channel dropout, time shift) were critical to prevent overfitting.
- Subject‑wise splitting is mandatory – epoch‑wise splits give overly optimistic results.
- EEG‑GAT did not reach the same performance, possibly because the static graph (based on 10‑20 adjacency) is too simplistic; future work could use data‑driven connectivity.

---

### License
This project is licensed under the MIT License – see the LICENSE file for details.

---

### Citation
If you use this code or results, please cite:

bibtex
@misc{alzheimers_eeg_2025,
  author = {BurhanxGodhra},
  title = {Alzheimer's EEG Binary Classifier: EEGNet, Transformer, and GAT for early detection},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/BurhanxGodhra/alzheimers-eeg-classifier}}
}

---

### Acknowledgements
- OpenNeuro for providing the DS007526 dataset.
- The authors of EEGNet, PyTorch, MNE, and PyTorch Geometric for their excellent libraries.
- Kaggle for free GPU compute during development.

---

This repository serves as a complete, reproducible pipeline for Alzheimer’s EEG classification. Future work will apply the same methodology to prodromal Parkinson’s disease detection using the TDBrain dataset – stay tuned.
