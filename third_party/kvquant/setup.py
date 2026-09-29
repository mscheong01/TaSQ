from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kmeans_tools',
    ext_modules=[
        CUDAExtension(
            name='kmeans_tools',
            sources=['kmeans_cuda/kmeans_tools.cpp',
                     'kmeans_cuda/dist_argmin_half_batched.cu',],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                         '--use_fast_math',
                         '-lineinfo']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
