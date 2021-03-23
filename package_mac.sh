#!/bin/bash
pyinstaller -F --paths $CONDA_PREFIX --add-data "static:static" --add-data "minerva-story:minerva-story" --add-data "$CONDA_PREFIX/lib/python3.7/site-packages/xmlschema/schemas:xmlschema/schemas" src/app.py
