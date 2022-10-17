#!/bin/bash
pyinstaller -F --paths $CONDA_PREFIX src/app.py
