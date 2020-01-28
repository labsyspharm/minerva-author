### Installing

```
conda create --name author python=3.6
pip install -r requirements.txt
```

### Packaging

```
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' src/app.py
```
