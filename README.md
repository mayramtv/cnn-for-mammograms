# Deep Learning Models for Breast Cancer Detection in Mammography

This project explores the use of deep learning models for breast cancer detection using mammography images from the CBIS-DDSM dataset. The goal was to compare preprocessing techniques and CNN-based architectures for classifying mammogram cases as benign or malignant.

The project includes a full experimentation workflow using:

* Custom CNN models
* VGG16 transfer learning
* ResNet50 transfer learning
* Image preprocessing experiments
* Model evaluation and comparison
* Final inference notebook for prediction

> **Disclaimer:** This project is for academic and research purposes only. It is not intended for clinical diagnosis or medical decision-making.

---

# Project Motivation

Breast cancer detection can be limited by access to specialists, imaging resources, and diagnostic infrastructure. This project investigates whether deep learning models can help flag mammography images for further review, especially in low-resource settings.

The project was developed as a final Computer Science (Data Science specialization) project and focuses on understanding the impact of preprocessing techniques, model architecture selection, and transfer learning on mammogram classification.

---

# Dataset

The project uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset from The Cancer Imaging Archive (TCIA).

Because the full DICOM image files are large, they are not included in this repository. The metadata CSV files are included, but the image files must be downloaded separately.

Dataset source:

https://www.cancerimagingarchive.net/collection/cbis-ddsm/

---

# Project Workflow

The project is organized into multiple phases:

## Phase 1 — Data Preparation

* Download CBIS-DDSM data
* Decompress DICOM images
* Match image paths with metadata
* Clean labels and prepare binary classification data

## Phase 2 — Baseline Model

* Build a basic Custom CNN
* Train on mammogram images
* Identify dataset and label issues
* Establish baseline performance

## Phase 3 — Preprocessing Experiments

Evaluate preprocessing techniques including:

* Background removal
* Cropping
* Noise reduction
* CLAHE contrast enhancement
* Edge enhancement
* Local Binary Pattern (LBP) texture extraction

## Phase 4 — Model Comparison

Compare:

* Custom CNN
* VGG16 Transfer Learning
* ResNet50 Transfer Learning

## Phase 5 — Final Inference

* Load the best saved model
* Run predictions on unseen mammogram images
* Evaluate deployment readiness

---

# Key Results

The best overall model was **VGG16 with basic preprocessing**.

Final model performance after threshold adjustment:

| Metric               | Result |
| -------------------- | ------ |
| Accuracy             | 62.8%  |
| Recall (Sensitivity) | 82.7%  |
| Specificity          | 45.7%  |

Recall was prioritized because, in a cancer detection context, missing malignant cases is more critical than flagging additional benign cases for review.

---

# Main Findings

* VGG16 produced the most balanced performance across metrics.
* Custom CNN models sometimes achieved high recall but very low specificity.
* ResNet50 showed more stable but lower overall performance.
* Several preprocessing techniques improved Custom CNN performance but did not outperform the final VGG16 model.
* Edge enhancement was the only preprocessing technique that consistently improved baseline CNN performance.
* Threshold adjustment improved recall but reduced specificity.

---

# Repository Structure

```text
cnn-for-mammograms/
├── CBIS-DDSM_Data/
├── CBIS-DDSM_Clean_Data/
├── Models/
├── Outputs/
├── Tests/
├── Utils/
├── P0_Decompress_Data.ipynb
├── P1_Basic_Preprocessing.ipynb
├── P1_Prototype_Model_1.ipynb
├── P1_Prototype_Model_2.ipynb
├── P1_Prototype_Model_3.ipynb
├── P2_0_Local_Preprocess.ipynb
├── P2_1_Iteration.ipynb
├── P2_2_Iteration.ipynb
├── P2_3_Iteration.ipynb
├── P2_3.2_Iteration.ipynb
├── P3_4_Iteration_Baseline.ipynb
├── P3_4_Iteration_EdgeEnh.ipynb
├── P3_5_Iteration.ipynb
├── P4_Final_Comparison.ipynb
├── P4_Inference.ipynb
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/mayramtv/cnn-for-mammograms.git
cd cnn-for-mammograms
```

## 2. Install Git LFS

The saved trained model uses Git LFS.

```bash
git lfs install
git lfs pull
```

## 3. Create a Virtual Environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch Jupyter Lab

```bash
pip install jupyterlab
jupyter lab
```

---

# Notebook Guide

| Notebook                      | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| P0_Decompress_Data.ipynb      | Decompress DICOM images and prepare image files |
| P1_Basic_Preprocessing.ipynb  | Prepare metadata and labels                     |
| P1_Prototype_Model_*.ipynb    | Build and test baseline CNN models              |
| P2_*_Iteration.ipynb          | Run preprocessing experiments                   |
| P3_4_Iteration_Baseline.ipynb | Compare Custom CNN, VGG16, and ResNet50         |
| P3_4_Iteration_EdgeEnh.ipynb  | Compare models using edge enhancement           |
| P3_5_Iteration.ipynb          | Fine-tune the selected model threshold          |
| P4_Final_Comparison.ipynb     | Compare final metrics                           |
| P4_Inference.ipynb            | Run inference using the saved best model        |

For prediction only:

```text
P4_Inference.ipynb
```

---

# Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* PyDicom
* PyWavelets
* Scikit-Image

### Data Processing

* NumPy
* Pandas
* Scikit-Learn

### Visualization

* Matplotlib

### Development

* Jupyter Notebook

---

# Future Improvements

Potential future work includes:

* Fine-tuning VGG16 layers instead of threshold-only tuning
* Adding EfficientNet and Vision Transformer architectures
* Grad-CAM explainability visualizations
* Patient-level train/validation/test splitting
* External validation datasets
* Hyperparameter optimization
* Better classifier head architecture
* Migration from notebook workflow to a modular Python package structure

---

# Limitations

This project is an academic deep learning experiment and should not be used for medical diagnosis.

The final model did not outperform published radiologist benchmark metrics. While recall approached reported radiologist sensitivity values, specificity remained significantly lower.

The project should be viewed as an experimentation framework for mammogram classification rather than a clinically deployable system.

---

# Author

**Mayra Torres**

Computer Science (Data Science Specialization)

Final Project: *Deep Learning Models for Breast Cancer Detection in Mammography*

University of London
