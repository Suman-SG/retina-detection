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

## Results

Metrics were computed from `evaluation_results.csv` (500 samples):

- **Accuracy:** 81.20%
- **Weighted precision:** 77.72%
- **Weighted recall:** 81.20%
- **Weighted F1-score:** 78.65%
- **Macro precision:** 57.84%
- **Macro recall:** 53.97%
- **Macro F1-score:** 54.68%

Per-class breakdown (as recorded in `evaluation_results.csv`):

- **Class 0** — support: 236 — precision: 95.88% — recall: 98.73% — F1: 97.29%
- **Class 1** — support: 45  — precision: 68.00% — recall: 37.78% — F1: 48.57%
- **Class 2** — support: 153 — precision: 69.74% — recall: 88.89% — F1: 78.16%
- **Class 3** — support: 21  — precision: 0.00%  — recall: 0.00%  — F1: 0.00%
- **Class 4** — support: 45  — precision: 55.56% — recall: 44.44% — F1: 49.38%

Note: class IDs correspond to the label encoding used in the dataset. If you use human-readable labels (for example: `0=healthy, 1=disease_A, ...`), update the table accordingly.

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
