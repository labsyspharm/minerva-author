### Installing

```
conda create --name author python=3.6
pip install -r requirements.txt
```

### Running

```
python src/app.py
```
Then open a browser to `localhost:2020`

### Packaging

```
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' src/app.py
```
