"""BuildFrame package initialization."""

import ctypes
import os
import sys
from pathlib import Path

__version__ = "0.2.0"


def _configure_windows_freeglut() -> None:
    """Bridge Conda's DLL name to the name expected by PyOpenGL on Windows."""
    if os.name != "nt":
        return
    dll_path = Path(sys.prefix) / "Library" / "bin" / "freeglut.dll"
    if not dll_path.exists():
        return
    try:
        from OpenGL import platform

        platform.PLATFORM.GLUT = ctypes.WinDLL(str(dll_path))
    except (ImportError, OSError):
        # The viewer handles an unavailable GLUT gracefully; point rendering
        # itself does not depend on it.
        return


_configure_windows_freeglut()
