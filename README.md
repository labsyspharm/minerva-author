### Preparing to install

#### On OSX

Run the following command to install libtiff.

```
brew install libtiff
```

#### On Windows

Clone the pytiff source.

```
git clone https://github.com/FZJ-INM1-BDA/pytiff.git
cd pytiff
```

Then [install pytiff from source](https://github.com/FZJ-INM1-BDA/pytiff#installation-on-windows).

### Installing
```
git clone git@github.com:labsyspharm/minerva-author.git
cd minerva-author
conda create --name author python=3.6
conda activate author
pip install numpy
pip install -r requirements.txt
```

### Running

```
python src/app.py
```

- Then, open a browser to `localhost:2020`

- Then, copy the filepath to an OME Tiff

- Wait for the generation of a full pyramid

At minimum, you'll need to type one 'Group' name into the top dropdown to create a group. For each group you create, you can select channels from the second dropdown and set up their rendering settings with the various sliders. After you hit 'save', look in the directory of the original OME Tiff for an `out.yaml` configuration file and an `out` directory of rendered images for use with Minerva Story.

### Packaging

#### MacOS

```
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' src/app.py
```

#### Windows powershell

```
pyinstaller -F --paths $env:CONDA_PREFIX --add-data 'static;static' src/app.py
```
