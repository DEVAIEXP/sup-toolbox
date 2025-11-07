import ast
import gc
import importlib
import inspect
import os
import platform
import secrets
import string
import sys
from enum import Enum
from pathlib import Path

import torch
from pyparsing import Any
from torch.nn import functional as F


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


cpu_state = CPUState.GPU
xpu_available = False
directml_enabled = False

try:
    if torch.xpu.is_available():
        xpu_available = True
except Exception:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except Exception:
    pass


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return True


def get_optimal_device_name():
    if torch.cuda.is_available():
        return "cuda"

    if has_mps():
        return "mps"

    return "cpu"


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())


def release_memory():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()


def open_folder(open_folder_path):
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


def infer_type(s: Any):
    """
    Infers and converts a string to the most likely data type.

    It attempts conversions in the following order:
    1. Python literal (list, dict, tuple, etc.) if the string looks like one.
    2. Integer
    3. Float
    4. Boolean (case-insensitive 'true' or 'false')
    If all conversions fail, it returns the original string.

    Args:
        s: The input value to be converted.

    Returns:
        The converted value or the original value.
    """
    if not isinstance(s, str):
        # If the input is not a string, return it as is.
        return s

    # 1. Try to evaluate as a Python literal (list, dict, etc.)
    s_stripped = s.strip()
    if s_stripped.startswith(("[", "{")) and s_stripped.endswith(("]", "}")):
        try:
            return ast.literal_eval(s_stripped)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            pass

    # 2. Try to convert to an integer
    try:
        return int(s_stripped)
    except ValueError:
        pass

    # 3. Try to convert to a float
    try:
        return float(s_stripped)
    except ValueError:
        pass

    # 4. Check for a boolean value
    s_lower = s_stripped.lower()
    if s_lower == "true":
        return True
    if s_lower == "false":
        return False

    # 5. If nothing else worked, return the original string (sem os espaços extras)
    return s


def random_code(size=6):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(size))


def get_dict_value(dict, key):
    if key in dict:
        return dict[key]
    else:
        return None


def get_module_file(module, file_name):
    try:
        module_path = inspect.getfile(module)
        module_dir = os.path.dirname(module_path)
        module_dir = Path(module_dir)
        
        #try on module folder
        file_path = os.path.join(module_dir, file_name)
        if not os.path.exists(file_path):
            print("File not found in this module.")
            pass
        else:
            return file_path
        
        #try on root module file
        file_path = os.path.join(module_dir.parent.parent, file_name)
        if not os.path.exists(file_path):            
            raise FileNotFoundError 

        return file_path

    except (TypeError, FileNotFoundError):
        print("File not found in this module.")
        return


def find_dist_info_files(package_name: str, file_name: str) -> Path | str:
    """
    Finds file within a package's .dist-info metadata directory.

    Args:
        package_name: The name of the package as you would `pip install` it.
        file_name: The name of the file as you wish get.
    Returns:
        A Path object to the license file, or an error string if not found.
    """
    try:
        # Get the list of all files included in the distribution
        dist_files = importlib.metadata.files(package_name)

        # Find the file in that list (case-insensitive search)
        file_obj = next((f for f in dist_files if file_name in f.name.upper()), None)

        if file_obj:
            # .locate() returns a Path object to the actual file on disk
            return file_obj.locate()
        else:
            return f"File not found in {package_name} metadata."

    except importlib.metadata.PackageNotFoundError:
        return f"Package '{package_name}' not found."
