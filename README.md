# Skin Cancer Detection (HAM10000)

This project builds an end-to-end image classification pipeline for **skin lesion diagnosis** using the **HAM10000** dataset (10,015 dermatoscopic images, **7 classes**). The goal is to predict the lesion type from an image while avoiding common pitfalls like train/test leakage across the same lesion.

## Key Features
- **Leakage-safe data splitting:** Uses **lesion-wise grouped splits** (`lesion_id`) to ensure images of the same lesion never appear in both train and test sets.
- **Computer vision preprocessing (OpenCV):**
  - HSV **gradient suppression** to reduce lighting/camera noise artifacts  
  - **k-means (k=2)** segmentation to isolate the lesion region  
  - Gaussian smoothing to down-weight background skin texture
- **Deep learning (TensorFlow/Keras):**
  - **Baseline:** Frozen **ResNet50 (ImageNet)** backbone + Global Average Pooling + softmax head
  - **Improved model:** Extracts CNN embeddings, applies **correlation-based feature filtering**, then trains a small MLP classifier with Early Stopping + LR scheduling
- **Evaluation:** Reports **accuracy, balanced accuracy, macro-F1**, confusion matrix, and per-class precision/recall.

## Dataset
- **HAM10000 (Skin Cancer MNIST):** 7 diagnostic categories  
  `akiec, bcc, bkl, df, mel, nv, vasc`  
- Split sizes (lesion-grouped): **7,121 train / 870 val / 2,024 test**

## Results (Held-out Test Set)
| Model | Accuracy | Macro-F1 | Balanced Acc |
|------|----------:|---------:|-------------:|
| ResNet50 baseline | 0.75 | 0.46 | 0.45 |
| Embeddings + filtered features + MLP | **0.79** | **0.60** | **0.57** |

## How to Run
1. Open the notebook: `4ML3_skin_cancer_detection.ipynb`
2. Install dependencies (example):
   ```bash
   pip install tensorflow opencv-python scikit-learn pandas numpy matplotlib kagglehub
