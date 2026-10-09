import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_geometry_polygon_sam_manual_ui(tmp_path):
    pytest.importorskip('rasterio');pytest.importorskip('pyproj')
    script=Path(__file__).with_name('geometry_ui_scenario.py')
    result=subprocess.run([sys.executable,str(script),str(tmp_path)],env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
