## Minimal PyInstaller Issue

Reproduce [this issue](https://github.com/pyinstaller/pyinstaller/issues/7165):

```
conda env create -f requirements.yml
conda activate minimal-issue
```

Run `bash package_mac.sh` or:

```
pyinstaller -F --paths $CONDA_PREFIX src/app.py
```
