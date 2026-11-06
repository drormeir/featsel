"""
Shared PyTorch plumbing for the selectors that can run on a GPU.

A long unattended run must not die of GPU memory. Three rules keep it alive:

- One fit at a time holds the GPU, across every process on the machine,
  through a file lock. Runner workers wait for it instead of crowding the
  GPU's memory.
- Batch sizes come from the GPU's free memory, not from a constant.
- A GPU out-of-memory error, which PyTorch and XGBoost raise as different
  types, becomes MemoryError. The runner already skips a cell on
  MemoryError and lists it at the end.

PyTorch is imported only when a GPU is asked for, so the package works
without it.
"""

import fcntl
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

_LOCK_FILE = Path(tempfile.gettempdir()) / 'featsel_gpu.lock'


def import_torch(device):
    """PyTorch, or an ImportError that names the device that needed it."""
    try:
        import torch
    except ImportError as error:
        # Under a tight address-space cap PyTorch is installed but cannot
        # map its libraries, so the message does not claim it is missing.
        raise ImportError(f"device={device!r} could not import PyTorch. "
                          "If it is missing: pip install -e '.[gpu]'") from error
    return torch


def cuda_usable():
    """True when PyTorch imports and a CUDA GPU is present."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve(device, X=None):
    """
    The device a fit runs on, as text.

    A tensor X on a GPU runs on its own device, whatever `device` says.
    'auto' is 'cuda' when PyTorch imports and a GPU is present, else 'cpu'.
    Any other device is returned as given.
    """
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(X, torch.Tensor) and X.device.type != 'cpu':
        return str(X.device)
    if device == 'auto':
        return 'cuda' if cuda_usable() else 'cpu'
    return device


def on_gpu(device):
    """True for a CUDA device, given as text ('cuda', 'cuda:0') or a torch.device."""
    return str(device).startswith('cuda')


@contextmanager
def exclusive(device, wait=True):
    """
    Hold the GPU alone for the length of the block; yield whether it is held.

    On a device other than CUDA there is nothing to hold and it yields True.
    With wait=False it yields False at once if another fit holds the GPU.

    Inside the block, a GPU out-of-memory error is raised as MemoryError.
    On leaving, PyTorch's cached GPU memory is released, so an idle process
    holds little. The kernel releases the lock if the process dies.
    """
    if not on_gpu(device):
        yield True
        return
    with open(_LOCK_FILE, 'w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        except Exception as error:
            # torch.OutOfMemoryError, torch.AcceleratorError and XGBoostError
            # all say this; none is a MemoryError.
            if not isinstance(error, MemoryError) and 'out of memory' in str(error):
                raise MemoryError(f"GPU out of memory on {device}: {error}") from error
            raise
        finally:
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def batch_size(torch, device, bytes_per_item, total):
    """
    How many items fit in half the device's free memory, between 1 and total.

    Half leaves room for PyTorch's own workspace and rounding to its block
    sizes. On the CPU, all items at once.
    """
    if not on_gpu(device):
        return total
    free, _ = torch.cuda.mem_get_info(device)
    return max(1, min(total, int(free // 2 // bytes_per_item)))
