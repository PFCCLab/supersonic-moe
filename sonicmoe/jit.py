# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import glob
import importlib.util
import inspect
import os
from shutil import rmtree
from typing import Callable
from uuid import uuid4

import torch
from filelock import FileLock


_CPP_MODULE_PREFIX = "sonicmoe"
# _GLOBAL_RANK = int(os.getenv("RANK", 0))
# _WORLD SIZE = int(os.getenv("WORLD_SIZE", 1))

_ALL_COMPILED_MODULES = {}


def _resolve_cpp_extension_load() -> Callable:
    """Resolve ``torch.utils.cpp_extension.load`` lazily.

    sonicmoe is consumed under two import orders:

      1. **Production** — ``paddle.compat.enable_torch_proxy(...)`` is called
         BEFORE ``import sonicmoe``. ``torch.utils.cpp_extension`` is then a
         proxy onto ``paddle.utils.cpp_extension`` and the returned ``load``
         produces a paddle-native ``_pd_.so`` that accepts ``paddle.Tensor``.

      2. **CI / warmup** — sonicmoe is imported first; the proxy is enabled
         later (e.g. inside ``warmup_jit``). If we cached ``load`` at
         module-import time we'd hold real torch's ``load`` forever, which
         JIT-compiles a torch-pybind ``.so`` whose pybind binding rejects
         ``paddle.Tensor`` with the misleading
         ``TypeError: deepep_topk_metadata_cuda(): incompatible function arguments``.

    Resolving lazily on each compile makes both paths work correctly.
    """
    from torch.utils.cpp_extension import load as _load
    return _load


def _try_import_prebuilt(module_name: str, build_directory: str):
    """Fast-path: re-use a successfully built extension on disk.

    Production model: every rank is its own process and they all share the
    same GPFS-backed ``build_directory``. Once any one rank has produced a
    valid ``<dir>/<name>/<name>.so``, all other ranks should import it
    directly instead of racing into paddle's ``load()`` (which wipes the
    build dir and recompiles, corrupting the rank that already finished).

    The ``.so`` is a PYBIND11 module — importing it via
    ``spec_from_file_location`` exposes ``m.def(...)`` symbols directly.

    Returns ``None`` when artifacts are absent or import fails (cold start,
    mid-rebuild, partial cleanup, ABI mismatch).
    """
    so_path = os.path.join(build_directory, module_name, f"{module_name}.so")
    if not os.path.exists(so_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


# @torch.compiler.disable
def _get_cpp_function(function_name: str, module_name: str, source_files: list[str], build_directory: str) -> Callable:
    module_name = f"{_CPP_MODULE_PREFIX}_{module_name}"

    if torch.distributed.is_initialized():
        _GLOBAL_RANK = torch.distributed.get_rank()
        _WORLD_SIZE = torch.distributed.get_world_size()
    else:
        _WORLD_SIZE = 1

    extra_cflags = ["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"]
    extra_cuda_cflags = ["-O3", "-lineinfo"]
    extra_include_paths = [
        os.path.dirname(__file__),  # sonicmoe/include
        os.path.dirname(os.path.dirname(__file__)) + "/cutlass/include",  # cutlass
        os.path.dirname(os.path.dirname(__file__)) + "/cutlass/tools/util/include",  # cutlass
    ]

    module = _ALL_COMPILED_MODULES.get(module_name, None)
    if module is not None:
        return getattr(module, function_name)

    # Stable lock path that survives a wipe of build_directory.
    # Using the parent ``build/`` dir means the lock inode is preserved even
    # if a sibling process clears ``build/<module>/`` mid-cycle.
    parent_dir = os.path.dirname(os.path.normpath(build_directory)) or "."
    os.makedirs(parent_dir, exist_ok=True)
    lock_path = os.path.join(parent_dir, f".{module_name}.lock")

    # Production model: one process per rank, exclusive ownership of
    # ``build_directory`` (or a shared dir populated by a single warmup
    # process before workers fork). FileLock guards the rare case where two
    # producers race to create the artifacts. The fast-path import after lock
    # acquisition handles the case where another process already built it.
    with FileLock(lock_path):
        # Re-arm the paddle/torch-proxy blockers (idempotent). The initial
        # ``import sonicmoe`` may have happened BEFORE the consumer called
        # ``paddle.enable_compat()``; in that order our hipify blocker is
        # a no-op because the proxy didn't exist yet. Re-running here
        # guarantees the blocker is live by the time paddle's
        # ``cpp_extension.load()`` looks up ``torch.utils.hipify``.
        try:
            from sonicmoe._quack_compat import install_quack_paddle_compat
            install_quack_paddle_compat()
        except Exception:
            pass

        mod = _try_import_prebuilt(module_name, build_directory)
        if mod is not None:
            _ALL_COMPILED_MODULES[module_name] = mod
            return getattr(mod, function_name)

        os.makedirs(build_directory, exist_ok=True)
        load_cpp_extension = _resolve_cpp_extension_load()
        try:
            mod = load_cpp_extension(
                module_name,
                sources=source_files,
                extra_cxx_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
                verbose=True,
            )
        except TypeError:
            # Paddle compat shim only accepts the older `extra_cflags`.
            mod = load_cpp_extension(
                module_name,
                sources=source_files,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
                verbose=True,
            )
        # paddle's load() returns a wrapper whose attributes lazily dlopen
        # a sibling ``.so``. Re-import via the on-disk fast-path so we get a
        # module object whose binary location is unambiguous and survives
        # any later cleanup of ``build_directory`` by another process.
        prebuilt = _try_import_prebuilt(module_name, build_directory)
        if prebuilt is not None:
            mod = prebuilt
        _ALL_COMPILED_MODULES[module_name] = mod
        return getattr(mod, function_name)


def cpp_jit(
    function_name: str | None = None,
    extra_source_files: list[str] = [],
    build_directory: str | None = None,
    depth: int = 0,
) -> Callable:
    """wrapper to compile C++/CUDA source code at runtime.

    Args:
        function_name (str | None, optional): name of the function to expose from the C++ file, the python function
            name should match the funcion name in the C++ file if this is not specified. Defaults to None.
        extra_source_files (list[str], optional): any extra files to use for compilation, by default it scans the
            directory of the python stub file. Defaults to [].
        build_directory (str | None, optional): directory in which to place the build artifacts. Defaults to None.
        depth (int, optional): number of times dirname is called to get the build path. Defaults to 2.

    Returns:
        Callable: returns the wrapped function that can be used to call the C++ functions from python
    """
    cpp_function = None
    args_spec = None

    source_files = []
    source_files.extend(extra_source_files)

    calling_filename = inspect.stack()[1].filename
    calling_directory = os.path.dirname(calling_filename)

    for dirname, _, filenames in os.walk(calling_directory):
        filenames = [os.path.join(dirname, f) for f in filenames]
        filenames = filter(lambda f: os.path.splitext(f)[1] in [".cu", ".cpp"], filenames)
        source_files.extend(filenames)

    if build_directory is None:
        module_name = calling_directory
        for _ in range(depth):
            module_name = os.path.dirname(module_name)
        module_name = os.path.basename(module_name)

        build_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", module_name)

    def _run(*args, **kwargs):
        nonlocal cpp_function

        if cpp_function is None:
            cpp_function = _get_cpp_function(
                function_name=_run.__name__,
                module_name=module_name,
                source_files=source_files,
                build_directory=build_directory,
            )

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return cpp_function(*full_args)

    def _wrapper(function: Callable) -> Callable:
        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        _run.__doc__ = function.__doc__
        _run.__name__ = function.__name__ if function_name is None else function_name
        _run.__signature__ = inspect.signature(function)

        return _run

    return _wrapper
