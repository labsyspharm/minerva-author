## Minerva Repository Structure

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Minerva Story
The GitHub Pages site build is stored at [minerva-story](https://github.com/labsyspharm/minerva-story). The source code for the minified bundle is stored at [minerva-browser](https://github.com/labsyspharm/minerva-browser).

### Minerva Author
The Python Flask server along with automated testing is stored at [minerva-author](https://github.com/labsyspharm/minerva-author). The React UI is stored at [minerva-author-ui](https://github.com/labsyspharm/minerva-author-ui)

## Using Minerva Author

### Installing

All commands should be run in "Terminal" on MacOS and "Anaconda Prompt" on Windows.

First, download this repository through the git command line:

```
git clone https://github.com/labsyspharm/minerva-author.git
```

**Windows**

 * [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
 * Install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
 * Move [openslide](https://openslide.org/download/#windows-binaries) "bin" directory to "minerva-author/src"
 * Run `conda install -c anaconda git`

**MacOS**

 * install [homebrew](https://brew.sh/) and run `brew install openslide`.
 * Install [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/)

Then run the following commands to set up the development environment:

```
cd minerva-author
git submodule update --init --recursive
conda config --add channels conda-forge
conda create --name author python=3.8 nomkl
conda activate author
pip install numpy
conda env update -f requirements.yml
conda install openslide
conda install scikit-image
conda install zarr
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

### Automated Releases

All pushes to master will update the current draft relase.

### Packaging

#### MacOS

To package the application as a standalone executable, run script:
```
bash package_mac.sh
```

#### Windows (powershell)

Fetch OpenSlide binaries from https://openslide.org/download/#windows-binaries and save the .dll files to /src. Then run script:
```
package_win.bat
```

### Docker
#### Using docker-compose

To run the application using docker-compose, edit the '/my/local/path' path in the docker-compose.yaml to a directory 
which either contains your images or to a directory upstream:
```
/my/local/path:/opt/data-volume
```
Navigate to the root directory of the minerva-author repository and then run:
```
docker-compose up
```
A Docker image should be built and run. If subsequent code changes are made use the following command to ensure 
changes are incorporated:
```
docker-compose up --build
```

#### Using docker build/run

Alternatively a Docker image can be built and run without using docker-compose.
To build the Docker image navigate to the root minerva-author directory (this will contain the Dockerfile).
Then run this command:

```
docker build -t minerva-author .
```

To run the image after being built, annotate the '/my/local/path' to the path where your images exist:
```
docker run -p 2020:2020 -v /my/local/path:/opt/data-volume minerva-author
```