 # SAVADHAN - Retina Detection

 Final archived snapshot of the SAVADHAN Retina Screening project.

 ## Summary

 SAVADHAN is an experimental toolkit for binary retinal image classification (disease / no disease). This repository contains training, inference, and a small Streamlit demo for quick evaluation and demonstration.

 Status: **Archived - final snapshot**

 ## Highlights

 - Small Streamlit demo: `binary/web/smart_app.py`
 - Training & evaluation scripts: `train.py`, `evaluate.py`, `predict.py`
 - Models and checkpoints: `models/` and `checkpoints/`

 ## Suggested repository name

 - `retina-detection` (or `savadhan-retina-detection` for branded name)

 ---
 ## Quick metrics (how to reproduce)

 Use the provided `evaluation_results.csv` at the repository root to compute common metrics.

 Example Python snippet that computes accuracy, per-class recall, and a simple summary:

 ```python
 import csv
 from collections import Counter
 from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

 rows = list(csv.DictReader(open('evaluation_results.csv', newline='')))
 y_true = [int(r['true_label']) for r in rows]
 y_pred = [int(r['predicted_label']) for r in rows]

 print('n =', len(rows))
 print('accuracy =', accuracy_score(y_true, y_pred))
 print('precision =', precision_score(y_true, y_pred, average='binary'))
 print('recall =', recall_score(y_true, y_pred, average='binary'))
 print('f1 =', f1_score(y_true, y_pred, average='binary'))

 from collections import Counter
 print('true_counts =', Counter(y_true))
 print('pred_counts =', Counter(y_pred))
 ```

 If you don't have `scikit-learn`, install it with `pip install scikit-learn`.

 ---
 ## How to run (local)

 1. Create a virtual environment (recommended):

 ```powershell
 cd path\to\retina_detection
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 pip install -r requirements.txt
 ```

 2. Run the Streamlit demo (quick demo using a saved model):

 ```powershell
 cd binary/web
 # set MODEL_PATH environment variable to a model .h5 file
 $env:MODEL_PATH = 'C:\Users\you\...\binary\models\quick_test_model.h5'
 streamlit run smart_app.py
 ```

 3. Run evaluation on saved predictions (example):

 ```powershell
 python evaluate.py --pred predictions.csv --truth ground_truth.csv
 ```

 4. Make predictions with `predict.py` (example):

 ```powershell
 python predict.py --model models/binary_fast.keras --input data/images/ --output out.csv
 ```

 5. Train (advanced): read `train.py` and `scripts/train_binary.py` for dataset and hyperparameter details.

 ---
 ## Notes on models and large files

 This snapshot contains model files in `models/` and `checkpoints/`. Many of these are large binary blobs (HDF5 / PyTorch). If you plan to publish the repository on GitHub, consider removing large model files or using Git LFS.

 To remove model files from the repo before pushing, delete them locally and add them to `.gitignore`.

 ---
 ## How to push this project to GitHub

 Option A - using GitHub CLI (recommended, interactive):

 ```powershell
 cd path\to\retina_detection
 git init
 git add .
 git commit -m "chore: final snapshot - archive project"
 git branch -M main
 gh repo create USERNAME/retina-detection --public --source=. --remote=origin --push
 ```

 Option B - manual on GitHub:

 1. Create a new repository on https://github.com (name: `retina-detection`).
 2. Then run:

 ```powershell
 git remote add origin https://github.com/USERNAME/retina-detection.git
 git push -u origin main
 ```

 If you want the repo to be private, choose `--private` when using `gh repo create` or select Private on the website.

 ---
 ## Final notes & goodbye

 This is an archival snapshot of the project. Thank you for the experiments and work here - goodbye, retina_detection. If you'd like I can create the GitHub repo and push from your machine now; tell me your GitHub username and whether you want the repo public or private.
