# Smart App (Streamlit)

Instructions to create a virtual environment and run the Streamlit app locally.

1. Create a virtual environment (Windows):

```powershell
C:/Users/shonu/AppData/Local/Programs/Python/Python313/python.exe -m venv .venv
```

2. Install dependencies:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. (Optional) Set a model path if your trained model is not in `../../models/quick_test_model.h5`:

```powershell
$env:MODEL_PATH = "C:\path\to\your\model.h5"
```

4. Run the Streamlit app:

```powershell
.venv\Scripts\python.exe -m streamlit run smart_app.py
```

Then open the URL printed by Streamlit (http://localhost:8501 by default).
