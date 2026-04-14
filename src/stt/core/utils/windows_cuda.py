"""Windows CUDA DLL setup utilities."""

import ctypes
import importlib.util
import os
from pathlib import Path
import sysconfig


_DLL_HANDLES: list[object] = []


def add_dll_dir(path: str) -> bool:
    """Add a directory to the DLL search PATH and register with OS."""
    if not path or not os.path.isdir(path):
        return False
    h = os.add_dll_directory(path)
    _DLL_HANDLES.append(h)  # keep handle alive
    os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")  # belt + suspenders
    return True


def default_nvidia_bin(pkg: str) -> str | None:
    """Get default NVIDIA binary path for a package."""
    spec = importlib.util.find_spec(pkg)
    if spec is None or spec.submodule_search_locations is None:
        return None
    base = next(iter(spec.submodule_search_locations), None)
    if not base:
        return None
    return str(Path(base) / "bin")


def default_site_packages_bin(*parts: str) -> str | None:
    """Get default site-packages binary path.
    
    Avoid hard-coded absolute paths; work for venv + system installs.
    """
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        base = paths.get(key)
        if not base:
            continue
        candidate = Path(base).joinpath(*parts)
        if candidate.is_dir():
            return str(candidate)
    return None


def setup_windows_cuda_dlls(
    *,
    cublas_bin: str | None = None,
    cudnn_bin: str | None = None,
    check: bool = False,
    force_load: bool = False,
) -> None:
    """Setup Windows CUDA DLL paths and optionally verify/load them.
    
    Args:
        cublas_bin: Optional override path to cublas binary directory
        cudnn_bin: Optional override path to cudnn binary directory
        check: Check if DLL files exist
        force_load: Force load DLL files (raises if missing)
    """
    if os.name != "nt":
        return

    # Fallbacks from the current interpreter's site-packages (venv/system).
    fallback_cublas = default_site_packages_bin("nvidia", "cublas", "bin")
    fallback_cudnn = default_site_packages_bin("nvidia", "cudnn", "bin")

    cublas_bin = cublas_bin or default_nvidia_bin("nvidia.cublas") or fallback_cublas
    cudnn_bin = cudnn_bin or default_nvidia_bin("nvidia.cudnn") or fallback_cudnn

    added = [
        ("cublas", add_dll_dir(cublas_bin), cublas_bin),
        ("cudnn", add_dll_dir(cudnn_bin), cudnn_bin),
    ]

    print("Added DLL dirs:")
    for name, ok, path in added:
        print(f"  {name}: {ok}  ({path})")

    if not check and not force_load:
        return

    cublas_dll = os.path.join(cublas_bin, "cublas64_12.dll") if cublas_bin else None
    cudnn_dll = os.path.join(cudnn_bin, "cudnn_ops64_9.dll") if cudnn_bin else None

    if check:
        if cublas_dll:
            print("Exists cublas64_12.dll:", os.path.exists(cublas_dll))
        if cudnn_dll:
            print("Exists cudnn_ops64_9.dll:", os.path.exists(cudnn_dll))

    if force_load:
        if not cublas_dll or not os.path.exists(cublas_dll):
            raise FileNotFoundError(f"Missing DLL: {cublas_dll}")
        if not cudnn_dll or not os.path.exists(cudnn_dll):
            raise FileNotFoundError(f"Missing DLL: {cudnn_dll}")

        try:
            ctypes.WinDLL(cublas_dll)
            print("WinDLL load OK: cublas64_12.dll")
        except OSError:
            print("WinDLL load FAILED: cublas64_12.dll")
            raise

        try:
            ctypes.WinDLL(cudnn_dll)
            print("WinDLL load OK: cudnn_ops64_9.dll")
        except OSError:
            print("WinDLL load FAILED: cudnn_ops64_9.dll")
            raise
