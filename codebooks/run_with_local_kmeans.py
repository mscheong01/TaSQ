#!/usr/bin/env python3
"""Run a KVQuant script against a locally-built kmeans_tools instead of the vendored .so.

The vendored `third_party/kvquant/kmeans_tools*.so` is built against one specific torch ABI. When
the checkout is shared between machines, a machine whose torch does not match cannot overwrite it
without breaking the others. This wrapper keeps the shared file untouched.

sys.path order alone is NOT enough: llama_simquant.py runs with cwd = the vendor dir, so the
vendored .so sits in sys.path[0] and wins over PYTHONPATH, and the editable install maps the name
there as well. So the local build is IMPORTED HERE FIRST -- once it is in sys.modules, every
later `import kmeans_tools` resolves to it no matter what the path says. The assert is the guard:
silently running against the wrong kernel is exactly the failure this exists to prevent.

    KMEANS_LOCAL=/path/to/local/kmeans_tools/build \
        python3 run_with_local_kmeans.py llama_simquant.py <args...>
"""
import os
import runpy
import sys

local = os.environ.get("KMEANS_LOCAL")
if not local:
    sys.exit("run_with_local_kmeans.py: set $KMEANS_LOCAL to the directory holding the built .so")

sys.path.insert(0, local)
import torch  # noqa: F401,E402  -- must precede the extension so its libtorch symbols resolve
import kmeans_tools  # noqa: E402

assert kmeans_tools.__file__.startswith(local), \
    f"wrong kmeans_tools: {kmeans_tools.__file__} (wanted one under {local})"
print(f"[kmeans] using {kmeans_tools.__file__}", flush=True)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
