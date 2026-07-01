# Rice Leaf Disease Detection

This is a Data Science Capstone Project designed to detect and classify three major rice leaf diseases: **Bacterial leaf blight**, **Brown spot**, and **Leaf smut** using deep learning.

The project contains a complete exploratory analysis, experiments with baseline convolutional networks, transfer learning using MobileNetV2, and an interactive Plotly Dash dashboard for visualization and diagnosis prediction.

---

## Project Structure

```
Rice-Leaf-Disease-Detection/
│
├── Bacterial leaf blight/     # Raw dataset images for Blight
├── Brown spot/                # Raw dataset images for Spot
├── Leaf smut/                 # Raw dataset images for Smut
│
├── Rice_Leaf_Disease_Detection.ipynb   # Jupyter Notebook containing full EDA & Model Training
├── dashboard.py               # Plotly Dash Dashboard application
├── assets/
│   └── style.css              # Custom styles for the Dash UI
│
├── .gitignore                 # Excluded directories (caches, checkpoints)
└── README.md                  # Project documentation (this file)
```

---

## Installation & Setup

To run the notebook and the interactive dashboard, ensure you have the required dependencies installed:

```bash
pip install tensorflow opencv-python pandas numpy matplotlib plotly dash dash-bootstrap-components scikit-learn pillow
```

### Running the Notebook
Open the Jupyter Notebook `Rice_Leaf_Disease_Detection.ipynb` to inspect the training code:
```bash
jupyter notebook Rice_Leaf_Disease_Detection.ipynb
```

### Running the Dashboard
Start the local Plotly Dash dashboard server:
```bash
python dashboard.py
```
This starts a local development server at **[http://127.0.0.1:8050/](http://127.0.0.1:8050/)** and automatically opens a browser tab.

---

## Models & Results Summary

We compared three different configurations trained over 10 epochs:

| Model Architecture | Training Accuracy | Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **Model 1: Baseline CNN (Unaugmented)** | 96.0% | 64.0% | 62.5% |
| **Model 2: Baseline CNN (Augmented)** | 85.0% | 82.0% | 83.3% |
| **Model 3: MobileNetV2 (Transfer Learning)** | 98.0% | 95.0% | **95.8%** |

### Key Findings:
- **Baseline CNN (Model 1)** suffered from extreme overfitting due to a small dataset (~120 images total), showing high training accuracy but low test performance.
- **Data Augmentation (Model 2)** successfully resolved overfitting by synthesizing varied leaf inputs (random rotation, zoom, shifting, and flips).
- **MobileNetV2 (Model 3)** achieved the highest test accuracy (**95.8%**), leveraging pre-trained ImageNet features for superior generalization.

---

## Dashboard Features

The custom agriculture-themed student dashboard includes the following sections:
- **Dashboard Overview**: Key metrics (total images, class count, best validation accuracy) and dataset split values.
- **Dataset Overview**: Detailed properties of the dataset folders.
- **Exploratory Data Analysis**: Side-by-side symptom grids and resolution distribution charts.
- **Model Performance**: Loss and accuracy comparative curves, test confusion matrix heatmap, and precision/recall reports.
- **Disease Prediction**: Upload tool to classify a leaf image, rendering the predicted disease, prediction confidence bar, and treatment recommendations.
- **Data Augmentation**: Visualization of physical leaf shifts, zooms, flips, and rotations.
- **Project Challenges**: Documentation of challenges (small dataset size, training time, overfitting) and engineering solutions used.
- **Project Summary**: Quick summary table of findings and future research paths.

---

## Developed Using
- Python
- TensorFlow / Keras
- OpenCV
- Plotly Dash
- Bootstrap (via Dash Bootstrap Components)
- Pandas & NumPy
