FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Fichier 11 — `.flake8`
```
[flake8]
max-line-length = 100
exclude = __pycache__, .git, notebooks, models, data
ignore = E402, W503
```

---

## Fichier 12 — `.gitignore`
```
__pycache__/
*.pyc
*.pyo
.env
models/*.pkl
models/*.png
models/*.csv
data/
.pytest_cache/
.DS_Store
*.egg-info/
dist/
build/
.ipynb_checkpoints/