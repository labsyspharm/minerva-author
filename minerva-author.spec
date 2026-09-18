# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
import sys
import tomllib

is_macos = sys.platform == 'darwin'
with open('pyproject.toml', 'rb') as f:
    pp_data = tomllib.load(f)
    version = pp_data['project']['version']

datas = []
binaries = []
datas += collect_data_files('minerva_author')
hiddenimports = ['imagecodecs._shared', 'imagecodecs._imcd', 'imagecodecs._shared_cython', 'imagecodecs.jpeg8_decode', 'numcodecs.blosc', 'numcodecs.compat_ext']
hiddenimports += collect_submodules('xsdata_pydantic_basemodel')
tmp_ret = collect_all('altair')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ome_types')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
if is_macos:
    datas += [('Credits.rtf', '.')]

a = Analysis(
    ['minerva-author.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if is_macos:
    extra_args = []
else:
    extra_args = [a.binaries, a.datas]

exe = EXE(
    pyz,
    a.scripts,
    *extra_args,
    [('u', None, 'OPTION')],
    exclude_binaries=is_macos,
    name='MinervaAuthor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.getenv('CODESIGN_IDENTITY', None),
    entitlements_file=None,
    icon='icon.png',
)

if is_macos:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='minerva-author',
    )
    app = BUNDLE(
        coll,
        name='MinervaAuthor.app',
        icon='icon.png',
        bundle_identifier='org.labsyspharm.Minerva',
        version=version,
    )
