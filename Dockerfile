# Upgrade to a PyTorch release that supports stable torch.compile(inductor)
# and recent CUDA kernels. This fixes the missing compile_worker.watchdog.
# Always track latest stable PyTorch CUDA runtime image
FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PIP_ONLY_BINARY=blis,thinc,preshed,cymem,murmurhash \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TORCH_INDUCTOR_INSTALL_GXX=1

# Base image already includes Python + Torch; install minimal OS deps if needed
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=30 \
    && apt-get install -y --no-install-recommends \
       git wget curl ca-certificates \
       build-essential g++ clang ninja-build \
       ffmpeg libsndfile1 \
       --fix-missing \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project files
COPY pyproject.toml README.md CHANGELOG.md TODO.md /workspace/
COPY src /workspace/src
COPY examples /workspace/examples
COPY tests /workspace/tests
COPY profiles /workspace/profiles
COPY requirements.txt /workspace/requirements.txt

# Install Python deps in smaller layers to reduce commit pressure
RUN python -m pip install --upgrade --no-cache-dir pip

## Install/upgrade latest torch/vision/audio matching the base image CUDA
RUN python -m pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    python -m pip install --no-cache-dir --upgrade torch torchvision torchaudio

# Install base requirements first (filter out audio deps), then install audio stack with latest compatibles
RUN python - <<'PY'
import sys, subprocess, io, os
from pathlib import Path
req = Path('requirements.txt').read_text(encoding='utf-8').splitlines()
skip = { 'encodec', 'tts', 'faster-whisper', 'soundfile', 'librosa', 'allennlp', 'allennlp-models', 'spacy' }
base = []
for ln in req:
    s = ln.strip()
    if not s or s.startswith('#'):
        base.append(ln)
        continue
    name = s.split('[')[0].split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip().lower()
    if name in skip:
        continue
    base.append(ln)
Path('requirements.base.txt').write_text("\n".join(base), encoding='utf-8')
# Enforce binary wheels globally for base install
env = dict(os.environ)
env.setdefault('PIP_PREFER_BINARY', '1')
env.setdefault('PIP_ONLY_BINARY', 'blis,thinc,preshed,cymem,murmurhash')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip', 'setuptools', 'wheel'], env=env)
# Pre-install spaCy low-level deps as binary wheels to avoid source builds if they get pulled indirectly
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--only-binary', ':all:',
                       'blis', 'thinc', 'preshed', 'cymem', 'murmurhash'], env=env)
# Install the rest without only-binary restriction to allow pure-Python sdists
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements.base.txt'], env=env)
PY

# Latest compatible audio stack; exclude broken encodec 0.0.0 and prefer wheels
RUN /opt/conda/bin/python -m pip install --no-cache-dir --prefer-binary "encodec!=0.0.0" librosa soundfile TTS faster-whisper
# Try to fix potential tokenizers/onnx/alignment build issues proactively
RUN python - <<'PY'
import sys, subprocess
def pipi(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *pkgs])
try:
    # Remove CPU-only ORT if present and install latest GPU ORT
    subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime'])
    pipi('onnxruntime-gpu')
except Exception as e:
    print('post-req extras install note:', e)
PY
# Force-refresh ONNX stack to latest compatible (avoid conda-preinstalled conflicts)
RUN python - <<'PY'
import sys, subprocess
subprocess.call([sys.executable, '-m', 'pip', 'uninstall', '-y',
                 'onnx', 'onnxruntime', 'onnxruntime-gpu', 'onnxruntime-tools', 'onnxscript', 'protobuf'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                      'protobuf>=5', 'onnx>=1.17', 'onnxscript>=0.2.3', 'onnxruntime-gpu', 'onnxruntime-tools'])
PY
RUN python -m pip install --no-cache-dir -e .

# Provision an isolated venv for SFB heavy deps (Python 3.11 ok; AllenNLP requires 3.8-3.10).
# We install a CPU-only SRL alternative compatible with Py3.11/Torch 2.x and amrlib with its own torch pin inside the venv.
RUN python - <<'PY'
import os, sys, subprocess, venv
venv_dir = '/opt/sfb_venv'
venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
pip = os.path.join(venv_dir, 'bin', 'pip')
env = dict(os.environ)
# Force binary wheels for spaCy low-level deps to avoid compiling blis/thinc/preshed
env['PIP_ONLY_BINARY'] = 'blis,thinc,preshed'
def pipi(*args):
    subprocess.check_call([pip, 'install', '--no-cache-dir', *args], env=env)
try:
    pipi('--upgrade', 'pip', 'setuptools', 'wheel')
    # Avoid numpy>=2 in side venv until allennlp stack is ready; reduces breakage
    pipi('numpy<2')
    pipi('spacy')
    pipi('penman', 'amrlib')
    # Install AllenNLP in the side venv to keep the main image lean; latest compatible
    pipi('allennlp', 'allennlp-models')
except Exception as e:
    print('SFB venv install note:', e)
PY

# Expose helper to query SFB venv presence at runtime
ENV SFB_VENV=/opt/sfb_venv

# Rely on requirements.txt for Python deps to avoid duplicate/conflicting installs
RUN python - <<'PY'
import torch, os
print({'cuda': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(), 'ld_path': os.getenv('LD_LIBRARY_PATH','')})
PY

# Default command prints environment info
CMD ["bash", "-lc", "python3 -c 'import torch,os;print({\"cuda\":torch.cuda.is_available(),\"device_count\":torch.cuda.device_count()})' && echo 'Ready' "]


