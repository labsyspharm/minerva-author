#!/bin/bash
pyinstaller -F --add-data "static:static" --add-data "minerva-story:minerva-story" --collect-all altair --collect-all xmlschema --collect-all ome_types src/app.py
