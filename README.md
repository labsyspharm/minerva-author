### Installing

```
conda create --name minerva-author python=3.7
pip install -r requirements.txt
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' src/app.py
```
