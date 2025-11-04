"""
Deprecated shim retained for back-compat in docs/examples. Prefer onnx_decode_generate.py.
This module now proxies to the maintained runner.
"""

from .onnx_decode_generate import main


if __name__ == "__main__":
    main()
