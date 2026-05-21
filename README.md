# Automated Dimensional Inspection & Defect Detection

An industrial-grade computer vision and metrology pipeline designed to automatically measure hardware dimensions and detect surface anomalies in real-time. 

## 🚀 Key Features

* **Dynamic Spatial Calibration:** Utilizes **ArUco Markers (DICT_4X4_50)** to continuously calculate pixel-to-millimeter ratios, ensuring accurate measurements regardless of camera distance.
* **Sub-Pixel Edge Detection:** Implements a custom 1D derivative interpolation algorithm (`calc_subpixel_offset`) to refine edge boundaries beyond standard pixel resolution.
* **Rotational Invariance:** Hardware parts can be placed at any angle. The system calculates the primary dimensions using `cv2.minAreaRect`.
* **Signal Stabilization:** Employs temporal smoothing (Moving Average via `collections.deque`) to eliminate hardware measurement flickering.
* **Deep Learning Anomaly Detection (PoC):** Features an unsupervised Convolutional Autoencoder built on a **ResNet50 backbone** (PyTorch) to generate MSE reconstruction error heatmaps for surface defect visualization.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Computer Vision:** OpenCV, NumPy
* **Deep Learning:** PyTorch, Torchvision, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## 📦 Dataset & Inference Instructions

To ensure a lightweight repository, the full training and test datasets are hosted externally. 

* **Benchmark Dataset:** This project evaluates model performance using the industrial **MVTec Anomaly Detection (MVTec AD)** screw dataset, which is a worldwide standard benchmark for unsupervised anomaly detection.
* **Full Dataset Download:** [Download Full MVTec Screw Dataset](https://drive.google.com/drive/folders/1Y5OQwsHLxDs0NnyvwId5my2gpIdjAb1d?usp=sharing)
* **Quick Inference Demo:** A minimal structure of the MVTec folder with a few sample images is included in `sample_data/`. You can run the entire inference loop and generate defect heatmaps in the Jupyter Notebook (`ad_2_autoencoder_resnet.ipynb`) out-of-the-box without downloading the full dataset.

## 🧠 Core Metrology Logic (Snippet)
Instead of relying on basic Canny edge detection, this system calculates sub-pixel offsets to maintain industrial tolerance limits:

```python
def calc_subpixel_offset(val_left, val_center, val_right):
    vl, vc, vr  = float(val_left), float(val_center), float(val_right)
    denominator = 2.0 * (vl - 2.0 * vc + vr)
    if denominator == 0:
        return 0.0
    return (vl - vr) / denominator

