# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()

datas = [
    (str(project_root / "NN_NSGA2_ActiveLearning_refactored.py"), "."),
    (str(project_root / "pareto_front_query.py"), "."),
    (str(project_root / "pareto_export_cft_cases.py"), "."),
    (str(project_root / "cfx_runner.py"), "."),
    (str(project_root / "GUI_DESKTOP_APP.md"), "."),
]

block_cipher = None


a = Analysis(
    ["impeller_app/__main__.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "cfx_runner",
        "impeller_app",
        "impeller_app.gui.main",
        "impeller_app.core.active_learning",
        "impeller_app.core.pareto",
        "impeller_app.runner.external",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImpellerWorkbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ImpellerWorkbench",
)
