from __future__ import annotations

"""
Tiny latent refiner quick trainers for image and audio latents.

Runs a short training loop and optionally exports ONNX refiner models:
  - Image: wraps training/flow_recon with --use_refiner and ONNX export
  - Audio: wraps training/audio_recon with --use_refiner and ONNX export

Usage:
  python -m omnicoder.tools.refiner_train --kind image --data examples/data/vq/images --steps 200 \
    --export_onnx weights/refiners/image_refiner.onnx

  python -m omnicoder.tools.refiner_train --kind audio --mel_dir data/mels --steps 200 \
    --export_onnx weights/refiners/audio_refiner.onnx
"""

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick trainer for tiny latent refiners (image/audio)")
    ap.add_argument('--kind', type=str, required=True, choices=['image','audio'])
    ap.add_argument('--data', type=str, default='')
    ap.add_argument('--mel_dir', type=str, default='')
    ap.add_argument('--wav_dir', type=str, default='')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--export_onnx', type=str, default='')
    args = ap.parse_args()

    if args.kind == 'image':
        # Delegate to flow_recon with minimal flags
        cmd = [sys.executable, '-m', 'omnicoder.training.flow_recon',
               '--data', args.data or 'examples/data/vq/images',
               '--steps', str(int(args.steps)), '--batch', str(int(args.batch)), '--device', args.device,
               '--use_refiner']
        if args.export_onnx:
            cmd += ['--export_refiner_onnx', args.export_onnx]
    else:
        # Audio refiner
        cmd = [sys.executable, '-m', 'omnicoder.training.audio_recon',
               '--steps', str(int(args.steps)), '--batch', str(int(args.batch)), '--device', args.device,
               '--use_refiner']
        if args.mel_dir:
            cmd += ['--mel_dir', args.mel_dir]
        if args.wav_dir:
            cmd += ['--wav_dir', args.wav_dir]
        if args.export_onnx:
            cmd += ['--export_refiner_onnx', args.export_onnx]
    import subprocess
    print('[refiner_train] running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == '__main__':
    main()


