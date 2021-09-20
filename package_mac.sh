#!/bin/bash
pyinstaller -F --paths $CONDA_PREFIX --add-data "static:static" --add-data "minerva-story:minerva-story" --add-data "$CONDA_PREFIX/lib/python3.8/site-packages/altair/vegalite/v4/schema:altair/vegalite/v4/schema" --add-data "$CONDA_PREFIX/lib/python3.8/site-packages/xmlschema/schemas:xmlschema/schemas" src/app.py
