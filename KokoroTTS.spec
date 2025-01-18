# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gradio_interface.py'],
    pathex=[],
    binaries=[],
    datas=[('F:\\ai\\Kokoro-TTS\\my_model', 'my_model'), ('F:\\ai\\Kokoro-TTS\\voices', 'voices'), ('F:\\ai\\Kokoro-TTS\\espeak', 'espeak')],
    hiddenimports=['phonemizer', 'phonemizer.backend', 'espeakng', 'transformers', 'gradio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KokoroTTS',
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
    codesign_identity=None,
    entitlements_file=None,
)
