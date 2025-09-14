# Breaking the Statistical Similarity Trap in Extreme Convection Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.09195-B31B1B.svg)](https://arxiv.org/abs/2509.09195)

---

This repository contains the official code for the research paper:  
**Breaking the Statistical Similarity Trap in Extreme Convection Detection**

The work introduces a novel deep learning architecture to deliver operationally skillful forecasts of extreme convective weather in the **Bay of Bengal region**. It highlights a critical flaw in standard evaluation metrics (the *Statistical Similarity Trap*) and reveals a new meteorological insight, the *IVT Paradox*.

📄 *Paper: !https://arxiv.org/abs/2509.09195.

---

## 📌 Project Overview
This project is organized as a series of **Python Scripts** (`.py` files) and **Google Colab notebooks** (`.ipynb` files). Each notebook is an independent, runnable component of the research pipeline, designed for maximum accessibility and reproducibility. The workflow covers the entire process from **data processing** to **model training and evaluation**.

---
## 🚀 Repository Structure & Workflow
This project is organized as a series of scripts and Google Colab notebooks that reflect the complete research lifecycle.

*1. Data Acquisition and Preprocessing* `(/scripts/ar_filter, /scripts/download_data, /scripts/preprocess_new_data, /scripts/prepare_multi_variable_dataset)`
The initial pipeline is a set of Python scripts that automate the entire data curation process:

`filter_ar_files.py`: Identifies relevant Atmospheric River (AR) events from the global catalog.

`download_*.py`: Downloads the corresponding ERA5 and Himawari satellite data.

`preprocess_*.py`: Creates the manifest, regrids all variables, normalizes the data

`prepare_*.py`: Splits the dataset into training, validation, and test sets, and computes normalization statistics.

*2. Model Training and Evaluation* (`/scripts/training`)
This directory contains the core experimental notebooks, organized by research question:

`/architecture`: The `Architecture_Study.ipynb` notebook trains and evaluates the four different deep learning architectures (U-Net, ResNet U-Net, etc.).

`/ablation`: The `Ablation_Study.ipynb` notebook runs the systematic 21-configuration study to test the importance of different meteorological variables.

`/mos_baseline`, `/rf_baseline`, `/svr_baseline`: Notebooks for training and evaluating the three traditional machine learning baselines to demonstrate the "Statistical Similarity Trap."

`/unet_final`: The `UNet_Training.ipynb` notebook contains the final U-Net training with smart sampling and weighted loss.

`/dart`: The `HiPEEF_Training.ipynb` notebook contains the final, two-stage pipeline, combining the Stage 1 forecast engine and the Stage 2 diagnostic engine.

*3. Results and Figure*s (`/visuals`)
The /visuals directory contains the final output figures used in the paper.


---

## 🚀 How to Run

This project is designed to run entirely within **Google Colab**, requiring **no local dependency installation**.

### 1. Data Setup
The complete dataset is available upon request. Once obtained, upload it to Google Drive using the following structure:
```
📂 My Drive/
└── 📂 research/
    └── 📂 statistical_similarity/
        └── 📂 data/
            ├── 📂 train/
            ├── 📂 val/
            └── 📂 test/
```

### 2. Running the Notebooks
1. Open any `.ipynb` notebook from this repository in Google Colab.  
2. Run the following cell to mount your Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the PROJECT_PATH variable at the top of the notebook, e.g.:

```python
PROJECT_PATH = Path('/content/drive/My Drive/AR_Downscaling')
```
Also make sure the other similar paths are exactly what you want them to be.

4. Run all cells sequentially (Runtime > Run all).
All required Python libraries are automatically installed inside the notebooks.

## Pre-trained Models
The final trained model weights are too large to host on GitHub but are available upon request.  

📩 Contact: **tanveer.munim@outlook.com** for a download link.  

Alternatively, you can retrain the model from scratch using:    
- `dart/HiPEEF_Training.ipynb`

This will fully reproduce the paper’s results.

Note: DART was originally named HiPEEF (Hierarchical Prediction of Extreme Events with Forecasts).

## 📜 Citation
If you find this work useful, please consider citing our paper.
**Paper:** [https://arxiv.org/abs/2509.09195](https://arxiv.org/abs/2509.09195)

```bibtex
@misc{munim2025breakingstatisticalsimilaritytrap,
      title={Breaking the Statistical Similarity Trap in Extreme Convection Detection}, 
      author={Md Tanveer Hossain Munim},
      year={2025},
      eprint={2509.09195},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2509.09195](https://arxiv.org/abs/2509.09195)}, 
}
```

