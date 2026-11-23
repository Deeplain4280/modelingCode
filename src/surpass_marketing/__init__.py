"""Surpass 目标用户精准营销建模包。"""
import os as _os

# 单线程运行：本机沙箱禁止创建进程/线程池所需的管道句柄，
# 同时保证结果可复现（HGB 的 OpenMP 线程数、BLAS 线程数均固定为 1）。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")

__version__ = "1.0.0"
__all__ = ["__version__"]
