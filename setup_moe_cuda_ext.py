from __future__ import annotations

import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def main() -> None:
    this_dir = Path(__file__).parent
    src_dir = this_dir / 'src' / 'omnicoder' / 'modeling' / 'kernels'
    cuda_sources = [
        str(src_dir / 'moe_cuda_ext.cu'),
        str(src_dir / 'moe_cuda_ext.cpp'),
    ]
    extra_compile_args = {
        'cxx': ['-O3'],
        'nvcc': ['-O3', '--use_fast_math']
    }
    setup(
        name='moe_cuda_ext',
        ext_modules=[
            CUDAExtension('moe_cuda_ext', sources=cuda_sources, extra_compile_args=extra_compile_args)
        ],
        cmdclass={'build_ext': BuildExtension}
    )


if __name__ == '__main__':
    main()


