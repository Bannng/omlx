# SPDX-License-Identifier: Apache-2.0
"""Temporary CI diagnostic: where a 4-row DSv4.1 verify and a 3-row replay part."""

import inspect
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_deepseek_v41 import load_reference_weights
from test_deepseek_v41_mtp import mtp_config

from omlx.patches.deepseek_v41 import language as L


def _np(x):
    if x.dtype in (mx.bfloat16, mx.float16):
        x = x.astype(mx.float32)
    return np.array(x)


def _prefix_rows(a, b):
    if a.shape == b.shape:
        return a, b
    if a.ndim != b.ndim:
        return None
    axes = [i for i, (x, y) in enumerate(zip(a.shape, b.shape)) if x != y]
    if len(axes) != 1 or a.shape[axes[0]] < b.shape[axes[0]]:
        return None
    sl = [slice(None)] * a.ndim
    sl[axes[0]] = slice(0, b.shape[axes[0]])
    return a[tuple(sl)], b


def _diff(x, y):
    if x.dtype.kind == "f":
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    else:
        d = np.abs(x.astype(np.int64) - y.astype(np.int64))
    return int((d > 0).sum()), float(d.max()) if d.size else 0.0


def test_diag_dsv41_rollback_rows():
    rec = {"tag": None, "a": [], "b": []}
    originals = {}

    def wrap(cls):
        if cls in originals:
            return
        orig = cls.__call__
        originals[cls] = orig

        def call(self, *a, **k):
            out = orig(self, *a, **k)
            tag = rec["tag"]
            if tag:
                outs = out if isinstance(out, (tuple, list)) else (out,)
                arrs = [o for o in outs if isinstance(o, mx.array)]
                try:
                    mx.eval(arrs)
                except ValueError:
                    return out
                rec[tag].append((cls.__name__, [_np(x) for x in arrs]))
            return out

        cls.__call__ = call

    for obj in list(vars(L).values()):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            wrap(obj)
    for cls in (nn.Linear, nn.QuantizedLinear, nn.RMSNorm, nn.Embedding):
        wrap(cls)
    report = [f"mlx {mx.__version__} {mx.device_info().get('architecture')} {mx.device_info().get('device_name')}"]
    try:
        prefix, accepted, ratio = 7, 2, 2
        config = replace(mtp_config(), compress_ratios=(0, ratio, ratio, 1, 1, 0, 0, 0))
        model = L.LanguageModel(config)
        load_reference_weights(model)
        model.configure_mtp(True, 3)
        prompt = mx.array([[3 + i % 20 for i in range(prefix)]])
        cache, expected = model.make_cache(), model.make_cache()
        model(prompt, cache=cache)
        model(prompt, cache=expected)
        block = mx.array([[21, 22, 23, 24]])
        rec["tag"] = "a"
        out = model(block, cache=cache, return_hidden=True, n_confirmed=1)
        mx.eval([o for o in (out if isinstance(out, tuple) else (out,)) if isinstance(o, mx.array)])
        rec["tag"] = "b"
        out = model(block[:, : accepted + 1], cache=expected, return_hidden=True)
        mx.eval([o for o in (out if isinstance(out, tuple) else (out,)) if isinstance(o, mx.array)])
        rec["tag"] = None
        report.append(f"calls a {len(rec['a'])} b {len(rec['b'])}")
        shown = 0
        for i, ((na, xa), (nb, xb)) in enumerate(zip(rec["a"], rec["b"])):
            if na != nb:
                report.append(f"call {i}: order differs {na} vs {nb}")
                break
            for j, (x, y) in enumerate(zip(xa, xb)):
                pr = _prefix_rows(x, y)
                if pr is None:
                    continue
                bad, worst = _diff(*pr)
                if bad:
                    report.append(f"call {i} {na} out {j} shape {pr[0].shape} {pr[0].dtype}: {bad} differ, max {worst:.3g}")
                    shown += 1
            if shown >= 30:
                break
        assert model.mtp_partial_rollback(cache, accepted, 3)
        for li, (a_c, w_c) in enumerate(zip(cache, expected)):
            for ai, (a, b) in enumerate(zip(a_c.cache, w_c.cache)):
                a, b = np.array(a), np.array(b)
                if a.shape != b.shape:
                    report.append(f"cache layer {li} arr {ai} shape {a.shape} vs {b.shape}")
                    continue
                bad, worst = _diff(a, b)
                if bad:
                    idx = np.argwhere(a != b)[:4].tolist()
                    report.append(f"cache layer {li} arr {ai} {a.dtype} {a.shape}: {bad} differ, max {worst:.3g} at {idx}")
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig
    pytest.fail("\n".join(report))
