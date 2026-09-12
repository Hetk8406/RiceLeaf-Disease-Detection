# Rice Leaf Disease Detection

This is a Data Science Capstone Project designed to detect and classify three major rice leaf diseases: **Bacterial leaf blight**, **Brown spot**, and **Leaf smut** using deep learning.

The project contains a complete exploratory analysis, experiments with baseline convolutional networks, transfer learning using MobileNetV2, and an interactive Plotly Dash dashboard for visualization and diagnosis prediction.

---

## Project Structure

```
DS-Project-05-Rice-Leaf-Disease-Detection/
│
├── Bacterial leaf blight/     # Raw dataset images for Blight
├── Brown spot/                # Raw dataset images for Spot
├── Leaf smut/                 # Raw dataset images for Smut
├── RiceLeaf Disease Detection/# Application Screenshots (DS5-Project-1.png to 8.png)
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

## Dashboard Screenshots

<p align="center">
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-1.png" alt="Dashboard Overview" width="90%"/>
  <br><em>Dashboard Overview Page</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-2.png" alt="Dataset Overview" width="90%"/>
  <br><em>Dataset Overview & Class Distribution</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-3.png" alt="Exploratory Data Analysis" width="90%"/>
  <br><em>Exploratory Data Analysis & Resolution Distribution Scatter</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-4.png" alt="Model Performance" width="90%"/>
  <br><em>Model Performance & Accuracy/Loss Curves</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-5.png" alt="Disease Prediction Lab" width="90%"/>
  <br><em>Disease Detection & Image Upload Lab</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-6.png" alt="Data Augmentation" width="90%"/>
  <br><em>Data Augmentation & Transformation Visualizer</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-7.png" alt="Project Challenges" width="90%"/>
  <br><em>Project Challenges & Engineering Solutions</em><br><br>
  <img src="RiceLeaf%20Disease%20Detection/DS5-Project-8.png" alt="Project Summary" width="90%"/>
  <br><em>Project Summary & Production Recommendations</em><br>
</p>

---

## Developed Using
- Python
- TensorFlow / Keras
- OpenCV
- Plotly Dash
- Bootstrap (via Dash Bootstrap Components)
- Pandas & NumPy
