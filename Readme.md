# Product–Patent Linkage Prediction

This repository contains the codebase for the project **Product–Patent Linkage Prediction**. The goal of this project is to build a machine learning pipeline that automatically predicts associations between commercial products and relevant patents using contrastive learning and adversarial hard negative mining.

---

## 📌 Overview

Linking products with their corresponding patents is a critical task for IP intelligence and innovation tracking, but it is often hindered by sparse data and unstructured sources. This project tackles the problem using a combination of:

- A dual-encoder contrastive learning framework
- An adversarial policy-gradient-based hard negative mining strategy
- LLM-based text extraction and summarization
- Custom graph splitting strategies for robust evaluation

---

## 🗃️ Dataset

The dataset is built using publicly available product–patent associations from **Honeywell's Virtual Patent Marking (VPM)** page. Additional metadata is sourced from:

- **Google Patents** (via BigQuery) for patent descriptions
- **Web scraping** and **LLM summarization** for product pages

---

## 🧠 Model Architecture

- **Text Encoder:** Pretrained PaECTER + attention pooling
- **Loss Function:** Supervised contrastive loss
- **Adversarial Sampling:** Learned policies to sample hard negatives via policy gradients
- **Training Strategy:** Multi-epoch curriculum with increasing hard negative difficulty

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Patent_product
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers (for web scraping):**
   ```bash
   playwright install
   ```

---

## 📊 Usage

### Data Collection

1. **Scrape Honeywell VPM page:**
   ```bash
   python src/honeywell_webscrapper.py
   ```

2. **Use BigQuery to extract patent Data:**
   ```bash
   src/bigQuery_patent.sql
   ```

3. **Summarize patent descriptions:**
   ```bash
   python src/summarize_patent.py
   ```

4. **Extract relevant url links for every product:**
   ```bash
   python src/product_links.py
   ```

5. **Filter links list:**
   ```bash
   python src/filter_links.py
   ```

6. **Scrape URL content to generate product Data:**
   ```bash
   python src/product_info.py
   ```

### Training

**Run the complete training pipeline:**
```bash
python src/train_model.py
```

The training script will:
- Load and preprocess the dataset
- Split data using graph-based strategies
- Train dual encoders with adversarial hard negative mining
- Generate evaluation metrics and visualizations
- Save trained models and embeddings

### Evaluation

The model is evaluated using multiple metrics:
- **Precision@k** (k=1,5,10,50)
- **Mean Reciprocal Rank (MRR)**
- **Mean Average Precision (MAP)**
- **ROC AUC** and **PR AUC**

Results are automatically saved to the `Images/` directory with visualization plots.

---

## 📁 Project Structure

```
Patent_product/
├── src/                           # Source code
│   ├── train_model.py            # Main training script
│   ├── classes.py                # Model architectures and datasets
│   ├── helpers.py                # Utility functions and evaluation metrics
│   ├── honeywell_webscrapper.py  # Web scraping for product data
│   ├── summarize_patent.py       # LLM-based patent summarization
│   ├── product_info.py           # Product information extraction
│   ├── filter_links.py           # Data filtering utilities
│   ├── bigQuery_patent.sql       # Patent data extraction query
│   └── cluster_run_files/        # HPC cluster run scripts
├── Data/                         # Dataset files
│   ├── honeywell_patents_products.tsv
│   ├── products.json
│   ├── patents.json
│   ├── pairs.json
│   └── *_embeddings.csv         # Generated embeddings
├── Images/                       # Generated visualizations
│   ├── *_evaluation_metrics.png
│   ├── policy_heatmap.png
│   └── graph_sparsity.png
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

---

## 🎯 Key Features

### Graph-Based Data Splitting
- **Bridge-first strategy:** Prioritizes graph connectivity in train/test splits
- **Small-components-first:** Alternative splitting approach for comparison
- **Distribution-aware:** Maintains balanced node and edge distributions

### Adversarial Hard Negative Mining
- **Policy gradient learning:** Learns to sample challenging negatives
- **Curriculum learning:** Progressively increases difficulty across epochs
- **Dual adversaries:** Separate policies for products and patents

### Robust Evaluation
- **Multiple metrics:** Comprehensive evaluation with ranking and classification metrics
- **Visualization:** Automatic generation of performance plots and heatmaps
- **Cross-validation:** Proper train/validation/test splits with graph awareness

---

## 🔧 Configuration

### Model Hyperparameters

```python
LEARNING_RATE = 1e-5          # Encoder learning rate
ADV_LEARNING_RATE = 1e-5      # Adversary learning rate
EMBEDDING_DIM = 4096          # Final embedding dimension
TEMPERATURE = 0.02            # Contrastive loss temperature
BATCH_SIZE = 64               # Training batch size
NUM_EPOCHS = 3                # Number of training epochs
MAX_LENGTH = 512              # Maximum token sequence length
```
