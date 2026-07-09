# Public Transportation Delay Modeling: Weather & Events Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XH_VlDWOv-Zc9WL6yZei1uA5JKpUTyFP?usp=sharing)

An end-to-end deep learning framework designed to analyze, process, and classify public transportation delays resulting from extreme weather patterns and urban special events. This project evaluates and benchmarks feedforward, sequential, and attention-based neural network topologies.

---

## 📈 Final Model Performance Metrics

The classification architectures were trained and verified using a stratified testing split across four critical performance vectors: **Accuracy**, **Precision**, **Recall**, and **F1-Score**. The custom **Enhanced BiLSTM + Attention Model** achieved the highest overall semantic stability and predictive clarity.

### Architectural Benchmarks:
| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **ANN Baseline** (TF-IDF Space) | 81.00% | 0.8700 | 0.8739 | 0.8719 |
| **BiLSTM Main** (Padded Sequences) | 80.67% | 0.8402 | **0.9122** | 0.8747 |
| **Enhanced Model** (GloVe + Attention) | **83.00%** | **0.8977** | 0.8694 | **0.8833** |

### Core Quantitative Takeaways:
* **Attention Precision Maxima:** Integrating a custom `SimpleAttention` layer pushed the network's Precision to **0.8977**, significantly cutting down false-positive delay alarms.
* **BiLSTM Recall Peak:** The pure recurrent BiLSTM network achieved a maximum Recall of **0.9122**, rendering it exceptionally sensitive at capturing long-term sequential delay patterns.

---

## 🛠️ Pipeline Architecture & Feature Engineering

### 1. Data Cleaning & Extraction
* **Dataset Management:** Ingested structural delay log entries (`public_transport_delays.csv`).
* **Label Polarization:** Dropped ambiguous mid-tier metrics to lock in a clean binary target state representing structural delays versus on-time transit streams.
* **Sequence Engineering:** Structured text descriptions into a 120-token integer matrix utilizing a localized 5,000-word vocabulary index tokenizer.

### 2. Semantic Weighting
* Mounted Stanford NLP's 100-dimensional Global Vectors for Word Representation (`glove.6B.100d.txt`).
* Initialized a non-trainable spatial embedding matrix to map dense context rules before running classification computations.

---

## 🔬 Evaluated Neural Network Architectures

### Scenario A: Baseline Dense ANN
A basic feedforward structure utilizing sparse text vectors passed across continuous `Dense` hidden layers (128 $\rightarrow$ 64 units) paired with regularized Dropout bounds ($30\%$).

### Scenario B: Sequential Bidirectional LSTM
A recurrent framework running a 64-unit BiLSTM layer over sequential input strings, preserving chronological traffic and weather reporting data in both forward and backward time steps.

### Scenario C: Enhanced Attention Network (Winner)
Our top configuration routing sequential inputs into the pre-trained GloVe embedding block. It computes historical sequence vectors through a BiLSTM channel, focuses on high-impact anomaly terms via a custom `SimpleAttention` layer, and processes outcomes via regularized drop layers ($40\%$) for high-precision classification.

---

## 🖼️ Diagnostic Visualization
The analytical workflow automatically builds and saves diagnostic evaluation plots directly to the root directory:
* `perbandingan_performa_skenario.png`: A multi-variable bar chart tracking performance changes across all model iterations.

---

## 🚀 How to Run the Workspace

### Prerequisites
Install the required deep learning and spatial processing dependencies:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn keras-tuner
