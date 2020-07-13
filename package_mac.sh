#!/bin/bash
pyinstaller -F --paths $CONDA_PREFIX --add-data 'static:static' --add-data 'minerva-story;minerva-story' src/app.py