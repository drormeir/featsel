"""
The machine, libraries and code version a run used, saved beside its results.

A timing is only comparable with another taken on the same machine and code,
so every result file gets a JSON companion that says which. Plots read both.
"""

import json
import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

_PACKAGES = ['numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'torch', 'xgboost']


def describe():
    """The machine, the library versions and the git commit, as a dict."""
    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu': _cpu_name(),
        'cpu_count': os.cpu_count(),
        'ram_gb': _ram_gb(),
        'packages': {name: _version(name) for name in _PACKAGES},
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain', '--untracked-files=no')),
    }
    gpu = _gpu()
    if gpu is not None:
        info['gpu'], info['gpu_memory_gb'] = gpu
    return info


def now():
    """The local time with its UTC offset, to the second, as ISO text."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, **details):
    """
    Write details, the time and describe() to a JSON file; return its path.

    details hold what the run itself chose, such as its config or arguments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {'written': now(),
               **details, 'environment': describe()}
    path.write_text(json.dumps(content, indent=2, default=str) + '\n')
    return path


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args):
    """A git command's output in this package's repository, or None outside one."""
    try:
        return subprocess.run(['git', *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu():
    """
    The first NVIDIA GPU's name and memory in GB, from nvidia-smi, or None.

    Asked of the driver, not of PyTorch: the process writing the JSON may
    never import PyTorch while its workers use the GPU.
    """
    try:
        line = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    name, mib = line.rsplit(',', 1)
    return name.strip(), round(int(mib) / 1024, 1)


def _cpu_name():
    """The CPU model from /proc/cpuinfo, or what platform reports elsewhere."""
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb():
    """Total RAM in GB, or None where /proc/meminfo does not exist."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                return round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return None
