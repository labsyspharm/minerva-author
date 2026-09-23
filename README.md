## Download and run Minerva Author

New users should start with our [pre-built Windows and MacOS applications](https://github.com/labsyspharm/minerva-author/releases/latest). More detailed [download and launch instructions](https://www.minerva.im/download.html) can be found at the Minerva website, in addition to [complete instructions and tutorials](https://www.minerva.im/usage/).

Minerva Author may also be run on all platforms including Linux through [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uvx minerva-author
```

## Information For Software Developers

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Project Structure

#### Minerva Story
The GitHub Pages site build is stored at [minerva-story](https://github.com/labsyspharm/minerva-story). The source code for the minified bundle is stored at [minerva-browser](https://github.com/labsyspharm/minerva-browser).

#### Minerva Author
The Python Flask server along with automated testing is stored at [minerva-author](https://github.com/labsyspharm/minerva-author). The React UI is stored at [minerva-author-ui](https://github.com/labsyspharm/minerva-author-ui)

### Installing From the Source Repository

All commands should be run in "Terminal" on MacOS or "PowerShell" on Windows.

First, download this repository with `git` at the command line (on Windows you will need to download and install git first):

```
git clone https://github.com/labsyspharm/minerva-author.git
```

Then install `uv` from https://docs.astral.sh/uv/getting-started/installation/ .

### Running

```
cd minerva-author
uv run minerva-author
```

- Browser window should open automatically, if not then open a browser to `localhost:2020`

- Browse or copy the file path to an OME-TIFF

- Click import, and wait for the generation of a full pyramid if the OME-TIFF does not already include one.

At minimum, you'll need to type one 'Group' name into the top dropdown to create a group. For each group you create, you can select channels from the second dropdown and set up their rendering settings with the various sliders. After you click 'Save', look in your Documents folder for a `.story.json` file which contains the story configuration. This file can be loaded later to continue editing. If you click 'Publish' then the compiled Minerva Story will also be located in a sub-folder in the same location. This sub-folder can be copied to any web host to share the story online.

### Test suite

The project contains automated tests using the pytest framework. To run the test suite, run `pytest` via `uv` from the top level of the project:

```
uv run pytest
```

### Packaging

To build and package the application as a standalone executable, run:
```
uv run minerva-author-build-app
```
This will build a macOS app bundle on macOS or a self-contained .exe on Windows.

### Versioning

The version number is tracked in `pyproject.toml`. For releases, use `uv version --bump` then `git commit` and `git tag vX.Y.Z -m vX.Y.Z`.

### Automated builds

- All Pull Requests will be built for both macOS and Windows as per the Packaging section above and attached to the linked GitHub Actions run. This makes it easy to ship test builds to end users.

- Pushes to the master branch tagged `vX.Y.Z` will trigger the automatic creation of a Release with the macOS and Windows builds attached. Note that the Release will appear before both builds are complete but the artifacts will be attached eventually when their respective build pipelines complete.
