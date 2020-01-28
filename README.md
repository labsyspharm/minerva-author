### Installing

```
conda create --name author python=3.6
pip install -r requirements.txt
```

### Running

```
python src/app.py
```

- Then, open a browser to `localhost:2020`

- Then, copy the filepath to an OME Tiff

- Wait for the generation of a full pyramid

After you hit 'save', look in the directory of the original OME Tiff for an `out.yaml` configuration file and an `out` directory of rendered images for use with Minerva Story.

### Packaging

```
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' src/app.py
```
