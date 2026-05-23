# SAVADHAN - Retina Detection

Professional snapshot of the SAVADHAN Retina Screening project.

## Project overview

SAVADHAN is a research and experimentation repository for binary retinal image classification (disease vs. no disease). The codebase includes data processing, training scripts, model checkpoints, evaluation utilities, and a small Streamlit demo for local inspection.

This repository is provided as-is for reproducibility and demonstration purposes.

## Highlights

- Demo application: `binary/web/smart_app.py` (Streamlit)
- Core scripts: `train.py`, `evaluate.py`, `predict.py`
- Evaluation outputs: `evaluation_results.csv`, `performance_summary.png`, `confusion_matrix.png`

## Quick example: computing metrics

Use the included `evaluation_results.csv` to compute standard classification metrics. Example:

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

print('true_counts =', Counter(y_true))
print('pred_counts =', Counter(y_pred))
```

Install scikit-learn if needed: `pip install scikit-learn`

## Running locally (recommended workflow)

1. Create and activate a virtual environment:

```powershell
cd C:\\path\\to\\retina_detection
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Start the Streamlit demo (uses a saved model):

```powershell
cd binary/web
# set MODEL_PATH to a valid model file
$env:MODEL_PATH = 'C:\\full\\path\\to\\binary\\models\\quick_test_model.h5'
streamlit run smart_app.py
```

3. Evaluate predictions (example):

```powershell
python evaluate.py --pred predictions.csv --truth ground_truth.csv
```

4. Run inference (example):

```powershell
python predict.py --model models/binary_fast.keras --input data/images/ --output out.csv
```

5. Training and advanced usage: review `train.py` and `scripts/train_binary.py` for dataset preparation and hyperparameters.

## Notes on large files

Model and checkpoint files are large — they are excluded from the repository using `.gitignore`. To publish models alongside the code, use Git LFS:

```powershell
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "chore: enable Git LFS for model files"
git push
```

## Contributing and license

This repository is intended as an archived snapshot. If you plan to fork or continue development, please open an issue or submit a pull request with proposed changes. Add a `LICENSE` file if you need a specific copyright or reuse policy.

## Contact and attribution

Maintainer: Suman-SG

If you want any changes to the repository metadata (description, topics, or visibility), I can update the README further or prepare a short `CONTRIBUTING.md` and `LICENSE` file.
