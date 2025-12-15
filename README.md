![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

# 🌈 RGB–NIR Multispectral Segmentation using K-Means + Elbow Method  

Unsupervised Image Segmentation on the EPFL RGB–NIR Dataset

This project performs *multispectral image segmentation* using **RGB + Near Infrared (NIR)** channels.  
The goal is to find the optimal number of clusters (K) for K-Means using the **Elbow Method** and then apply that K for consistent segmentation across the dataset.

---

## 📌 Key Features  
- Uses **RGB + NIR** images from the EPFL RGB–NIR dataset  
- Extracts **per-pixel spectral features**  
  - R, G, B  
  - NIR  
  - N/R ratio  
  - N-R difference  
  - Brightness  
- Runs **K-Means for K = 2 to 8**  
- Uses **distance-to-line heuristic** to find the Elbow point  
- Aggregates per-image results to find **GLOBAL_K**  
- Performs final segmentation using GLOBAL_K  
- Saves output RGB, NIR, and Segmented PNGs  
- Fully automated pipeline and GitHub-ready script  

---

## 🧠 Project Pipeline

### **1️⃣ Per-Image Elbow Analysis**
For each RGB+NIR pair:
- Build feature vectors  
- Run K-Means for K ∈ [2..8]  
- Compute inertia  
- Calculate “distance to elbow line”  
- Select best K per image  
- Save elbow plots  

### **2️⃣ Global K Selection**
- Count frequency of best Ks  
- Plot histogram  
- Select **GLOBAL_K = most common K**  
- (In our experiment: GLOBAL_K = 4)

### **3️⃣ Final Segmentation**
- Run K-Means using GLOBAL_K on all images  
- Sort clusters by NIR reflectance (stable color assignment)  
- Save:  
  - `<name>_rgb.png`  
  - `<name>_nir.png`  
  - `<name>_segmented_K4.png`  

---

## 🛠 Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn (KMeans)  

## ▶️ How to Run

### **1. Install dependencies**
```bash
pip install numpy opencv-python scikit-learn matplotlib


