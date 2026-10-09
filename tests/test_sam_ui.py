import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_sam_mock_end_to_end(tmp_path):
    pytest.importorskip("rasterio");pytest.importorskip("pyproj")
    scenario = Path(__file__).with_name("sam_ui_scenario.py")
    result = subprocess.run([sys.executable, str(scenario), str(tmp_path)], env={**os.environ, 'QT_QPA_PLATFORM':'offscreen', 'PYTHONIOENCODING':'utf-8'}, capture_output=True, text=True, encoding='utf-8', timeout=100)
    assert result.returncode == 0, result.stdout+result.stderr
