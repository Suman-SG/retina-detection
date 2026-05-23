# Retina Disease Detection using Deep Learning

## Overview

This project presents a deep learning–based retinal disease detection system designed for binary classification of retinal fundus images.

The system classifies retinal scans into:

- Diseased Retina
- Healthy Retina

The repository includes:

- Model training pipeline
- Evaluation utilities
- Prediction scripts
- Performance visualization
- Streamlit-based web interface for testing

The objective of this project is to demonstrate the application of computer vision and deep learning techniques in automated medical image analysis.

## Features

- Binary retinal image classification
- Deep learning model training and evaluation
- Streamlit web application for prediction
- Confusion matrix and performance visualization
- CSV-based prediction analysis
- Modular Python implementation

## Project Structure

```
retina-detection/
│
├── binary/web/              # Streamlit web application
├── train.py                 # Model training script
├── evaluate.py              # Evaluation script
├── predict.py               # Prediction script
├── model.py                 # Model architecture
├── utils.py                 # Helper utilities
├── requirements.txt         # Dependencies
│
├── evaluation_results.csv
├── confusion_matrix.png
├── performance_summary.png
└── README.md
```

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/Suman-SG/retina-detection.git
cd retina-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Web Application

```bash
cd binary/web
streamlit run smart_app.py
```

## Model Evaluation

Example evaluation command:

```bash
python evaluate.py
```

Generated outputs include:

- Accuracy metrics
- Precision / Recall / F1-score
- Confusion matrix
- Performance graphs

## Sample Results

The project includes generated evaluation artifacts:

- `confusion_matrix.png`
- `performance_summary.png`
- `evaluation_results.csv`

These files provide insight into model performance and prediction quality.

## Future Improvements

- Multi-class retinal disease classification
- Improved dataset balancing
- Real-time deployment support
- Cloud-based inference API
- Explainable AI visualizations (Grad-CAM)

## Disclaimer

This project is intended for educational and research purposes only.
It is not a certified medical diagnostic system.

## Author

**Suman Ghosh**
GitHub: https://github.com/Suman-SG

## License

This project is open-source and available under the MIT License.

---
If you'd like, I can also add a short `CONTRIBUTING.md` and a `LICENSE` file, and push everything to the repo.
