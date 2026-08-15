# 🏥 Trial-Aware GCL Pipeline

**Heterogeneous Graph Contrastive Learning for Patient-Trial Matching**

A complete pipeline for learning patient representations that align with clinical trial eligibility criteria, using MIMIC-III data and real ClinicalTrials.gov trials.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Pipeline Architecture](#%EF%B8%8F-pipeline-architecture)
3. [Installation](#-installation)
4. [Data Requirements](#-data-requirements)
5. [Quick Start](#-quick-start)
6. [Configuration](#%EF%B8%8F-configuration)
7. [Evaluation](#-evaluation)
8. [Troubleshooting](#-troubleshooting)
9. [File Structure](#-file-structure)
10. [Citation](#-citation)

---

## 📌 Overview

This pipeline learns **trial-aware patient embeddings** by combining:
- **Self-supervised contrastive learning** (Stage A) on MIMIC-III patient data
- **Trial-aware alignment** (Stage B) using structured eligibility criteria from ClinicalTrials.gov

### Key Innovation

The model learns **bifurcated trial embeddings** (`z_inc` for inclusion, `z_exc` for exclusion) and aligns patient embeddings to them via a **dual-force alignment loss**.

### Published Results

| Metric | Stage A (Baseline) | Stage B (Full Model) | Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.4709 | **0.5533** | **+17.49%** |
| **PR-AUC** | 0.0190 | **0.0210** | **+10.07%** |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ MIMIC-III    │───▶│ Preprocessor │───▶│ Graph Constructor       │  │
│  │ (Raw CSVs)   │    │ (Phase 1)    │    │ (Phase 2)               │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────────┘  │
│                                                      │                  │
│  ┌──────────────┐    ┌──────────────┐               │                  │
│  │ ClinicalTrials│───▶│ Trial Store  │───────────────┘                  │
│  │ .gov (JSON)   │    │ (Parsing)    │                                   │
│  └──────────────┘    └──────────────┘                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ HeteroData Graph (HeteroGNNEncoder)                                ││
│  │ • Node Types: patient, diagnosis, medication, lab, trial           ││
│  │ • Edge Types: exhibits, prescribed_for, undergoes, matches         ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                       TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Stage A: Contrastive Pretraining (NT-Xent Loss)                   ││
│  │ • Graph augmentation with edge dropping (drop_rate=0.15-0.20)    ││
│  │ • Patient embeddings learned via self-supervision                ││
│  │ • Output: Baseline patient embeddings                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Stage B: Trial-Aware Alignment (Dual-Force Loss)                   ││
│  │ • Attraction: M_inc * D(z_P, z_T^inc)^2                           ││
│  │ • Hard-Neg Repulsion: λ₁ * M_exc * max(0, m_hard - D(z_P, z_T^exc))²││
│  │ • Rand-Neg Repulsion: λ₂ * max(0, m_rand - D(z_P, z_T^inc))²    ││
│  │ • Bifurcated trial embeddings (z_inc, z_exc)                     ││
│  │ • Anchor Loss: MSE(h_current, h_baseline) * λ_anchor            ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                       EVALUATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Evaluation Metrics                                                 ││
│  │ • ROC-AUC: Full Model vs Baseline                                 ││
│  │ • PR-AUC: Precision-Recall AUC                                    ││
│  │ • Improvement %: Stage B vs Stage A                               ││
│  │ • Comparison: Stage A (baseline) vs Stage B (trial-aware)        ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
| :--- | :--- |
| `preprocessor.py` | Clean MIMIC-III data, map ICD-9→ICD-10, expand lab trajectories |
| `graph_constructor.py` | Build heterogeneous graph with patient, diagnosis, medication, lab, trial nodes |
| `gcl_framework.py` | HeteroGNNEncoder + Contrastive Learning (NT-Xent) |
| `trial_graph.py` | Parse and store trial eligibility criteria |
| `trial_embedding.py` | Generate bifurcated trial embeddings (`z_inc`, `z_exc`) |
| `alignment.py` | Dual-force alignment loss for Stage B |
| `train.py` | Orchestrate Stage A + Stage B training |
| `evaluate.py` | Compute ROC-AUC, PR-AUC, and improvement metrics |

---

## 💻 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Trial_Aware_2
```

### 2. Create Conda Environment
```bash
conda create -n trial_aware python=3.11
conda activate trial_aware
```

### 3. Install Dependencies
```bash
pip install torch torch_geometric pandas numpy pyarrow tqdm scikit-learn
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## 📁 Data Requirements

### MIMIC-III Data
Place raw CSV files in your `DATA_DIR`:
```text
DATA_DIR/
├── ADMISSIONS.csv
├── DIAGNOSES_ICD.csv
├── PRESCRIPTIONS.csv
└── LABEVENTS.csv
```

### Clinical Trial Data

#### Option 1: Download from ClinicalTrials.gov
1. Go to [ClinicalTrials.gov](https://clinicaltrials.gov/ct2/results/download)
2. Select **JSON** format
3. Download **Top 1000** (or filter by conditions)
4. Save as `ctg-studies_1000.json`

#### Option 2: Use the Provided Script
```bash
# Place ctg-studies_1000.json in the current directory, then run:
python load_1000_trials.py
```
This will filter trials for target conditions and save them to `data/1000_trials/`.

### Target Conditions
For optimal performance with MIMIC-III data, filter trials for:
> *heart failure, myocardial infarction, diabetes, pneumonia, sepsis, atrial fibrillation, hypertension, chronic kidney disease*

---

## 🚀 Quick Start

### Run the Full Pipeline (SLURM)
```bash
sbatch job_pipeline.sh
```

### Run Steps Individually

1. **Preprocess MIMIC-III + Build Graph:**
   ```bash
   python run.py
   ```

2. **Train Model (Stage A + Stage B):**
   ```bash
   python train.py
   ```

3. **Evaluate Results:**
   ```bash
   python evaluate.py
   ```

### Expected Output Directory Structure
```text
data/
├── admissions_clean.parquet
├── diagnoses_clean.parquet
├── prescriptions_clean.parquet
├── labs_clean.parquet
├── hetero_graph.pt
├── patient_embeddings_baseline.pt   # Stage A embeddings
├── patient_embeddings.pt             # Stage B embeddings
├── trial_embeddings.pt               # Bifurcated trial embeddings
├── training_loss_history.csv
└── evaluation_results.json
```

---

## ⚙️ Configuration

### Key Parameters (`config.py`)

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `OUT_CHANNELS` | `32` | Embedding dimension |
| `EPOCHS_ALIGN` | `100` | Stage B epochs |
| `EPOCHS_GCL` | `100` | Stage A epochs |
| `LAMBDA_ANCHOR` | `0.5` | Anchor regularization weight |
| `LAMBDA_1` | `1.0` | Hard negative repulsion weight |
| `LAMBDA_2` | `2.5` | Random negative repulsion weight |
| `MARGIN_HARD` | `0.4` | Hard negative margin |
| `MARGIN_RAND` | `0.2` | Random negative margin |
| `HARD_NEG_INC_THRESHOLD` | `0.05` | Inclusion threshold for hard negatives |
| `HARD_NEG_EXC_THRESHOLD` | `0.8` | Exclusion threshold for hard negatives |

---

## 📊 Evaluation & Metrics

### Computed Metrics
- **ROC-AUC**: Area under Receiver Operating Characteristic curve
- **PR-AUC**: Area under Precision-Recall curve
- **Improvement %**: Relative gain from Stage B over Stage A

### Metric Interpretation Guide

| ROC-AUC Range | Performance Assessment |
| :---: | :--- |
| **0.50** | Random guessing baseline |
| **0.55 – 0.60** | Slight signal detected |
| **0.60 – 0.70** | Moderate alignment |
| **0.70 – 0.80** | Strong alignment performance |
| **0.80+** | State-of-the-art / Target for Q1 publication |

---

## 🔧 Troubleshooting

<details>
<summary><b>1. "No trials matched conditions"</b></summary>

**Fix:** Ensure your downloaded trials contain target MIMIC-III conditions. Download trials specifically for:
`heart failure`, `myocardial infarction`, `diabetes`, `pneumonia`, `sepsis`, `atrial fibrillation`, `hypertension`, `chronic kidney disease`.
</details>

<details>
<summary><b>2. "Anchor Loss = 0.0000"</b></summary>

**Fix:** The encoder learning rate is too low. In `train.py`, increase the learning rate multiplier:
```python
# Change from:
{'params': encoder.parameters(), 'lr': cfg.ALIGN_LR * 0.1}

# To:
{'params': encoder.parameters(), 'lr': cfg.ALIGN_LR * 0.5}
```
</details>

<details>
<summary><b>3. "CUDA Out of Memory"</b></summary>

**Fix:** Reduce batch size in `config.py` or switch execution to CPU:
```python
BATCH_SIZE = 128  # Reduce from 256
DEVICE = torch.device('cpu')
```
</details>

<details>
<summary><b>4. "z_inc == z_exc (cosine sim = 1.0)"</b></summary>

**Fix:** Include explicit separation loss in `trial_embedding.py`:
```python
sep_loss = F.cosine_similarity(z_inc, z_exc, dim=-1).mean()
```
</details>

---

## 📁 File Structure

```text
Trial_Aware_2/
├── config.py                    # Global Configuration & Hyperparameters
├── run.py                       # Pipeline Entrypoint: Preprocessing + Graph Construction
├── train.py                     # Training Engine: Stage A (GCL) + Stage B (Alignment)
├── evaluate.py                  # Evaluation Engine (ROC-AUC, PR-AUC)
├── generate_trial_json.py       # Trial Data Structuring
├── load_1000_trials.py          # Clinical Trial Downloader & Filter
├── preprocessor.py              # MIMIC-III Data Cleaning & ICD Mapping
├── graph_constructor.py         # PyTorch Geometric HeteroData Construction
├── gcl_framework.py             # HeteroGNNEncoder Architecture & NT-Xent Loss
├── trial_graph.py               # Eligibility Criteria Graph Structures
├── trial_embedding.py           # Bifurcated Embedding Generator
├── alignment.py                 # Dual-Force Alignment Loss Implementation
├── ontology_loader.py           # Medical Concept Mapping Tools
├── clinical_trials_api.py       # ClinicalTrials.gov API Connector
├── criteria_parser.py           # Eligibility Criteria NLP Parser
├── data/              # Data Output Directory
│   └── 1000_trials/             # Parsed & Structured Trial Data
├── job_pipeline.sh              # SLURM Master Job Script
└── README.md                    # Project Documentation
```

---

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{trial_aware_gcl_2026,
  title={Trial-Aware Graph Contrastive Learning for Patient-Trial Matching},
  author={Al Musawi, A.},
  journal={Under Review},
  year={2026}
}
```

---

## 📧 Contact & License

- **Author:** [Dr. Ahmad F. Al Musawi](https://almusawiaf.github.io/)
- **Project:** Research — Trial-Aware Patient Representation Learning
- **License:** For academic and research purposes only. MIMIC-III access required.
