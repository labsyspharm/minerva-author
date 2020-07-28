### Installing
```
git clone git@github.com:labsyspharm/minerva-author.git
cd minerva-author
conda config --add channels conda-forge
conda create --name author python=3.6 nomkl
conda activate author
pip install numpy
pip install -r requirements.txt
```

### Running

```
python src/app.py
```

- Browser window should open automatically, if not then open a browser to `localhost:2020`

- Browse or copy the file path to an OME-TIFF or SVS

- Click import and wait for the generation of a full pyramid

At minimum, you'll need to type one 'Group' name into the top dropdown to create a group. For each group you create, you can select channels from the second dropdown and set up their rendering settings with the various sliders. After you hit 'save', look in the directory of the executable (or app.py) for a new folder which contains the generated Minerva Story, with configuration files and an image pyramid.

### Automated test suite

The project contains automated tests using the pytest framework. To run the test suite, simply execute in the project folder:
```
pytest
```

### Packaging

#### MacOS

To package the application as a standalone executable, run script:
```
package_mac.sh
```

#### Windows (powershell)

Fetch OpenSlide binaries from https://openslide.org/download/#windows-binaries and save the .dll files to /src. Then run script:
```
package_win.bat
```

