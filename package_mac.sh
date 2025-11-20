#!/bin/bash
pyinstaller -F --add-data "static:static" --add-data "minerva-story:minerva-story" --collect-all altair --collect-all xmlschema --collect-all ome_types --collect-submodules xsdata_pydantic_basemodel --hidden-import "imagecodecs._shared" --hidden-import "imagecodecs._imcd" --hidden-import "imagecodecs.jpeg8_decode" src/app.py
