"""Run model-written pandas code in a separate process with a timeout.

Isolation level (be honest about it): separate process, wall-clock timeout, temp working dir,
whitelisted imports and no open/exec/eval/compile for the generated code. This is good enough for
a demo on a trusted local dataset but is NOT a security boundary - production would use a container
or a hosted sandbox.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = r'''
import ast, builtins, contextlib, io, json, sys, traceback
payload = json.load(sys.stdin)
import pandas as pd, numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
df = pd.read_json(io.StringIO(payload["df"]), orient="split")
ALLOWED = {"pandas","numpy","math","statistics","datetime","re","json","collections","itertools","functools",
           "decimal","fractions","matplotlib","scipy","calendar","string","textwrap","operator","time"}
BLOCKED_BUILTINS = {"open","exec","eval","compile","input","breakpoint","exit","quit","help"}
_real_import = builtins.__import__
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if globals and globals.get("__name__") == "__sandbox__" and name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of '{name}' is not allowed in the sandbox (allowed: {sorted(ALLOWED)})")
    return _real_import(name, globals, locals, fromlist, level)
safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
safe_builtins["__import__"] = safe_import
ns = {"__name__": "__sandbox__", "__builtins__": safe_builtins, "pd": pd, "np": np, "plt": plt, "df": df}
code = payload["code"]
buf = io.StringIO()
err = None
try:
    tree = ast.parse(code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<agent_code>", "exec"), ns)
        if last is not None:
            val = eval(compile(ast.Expression(last.value), "<agent_code>", "eval"), ns)
            if val is not None:
                print(val if isinstance(val, (pd.DataFrame, pd.Series)) else repr(val))
except BaseException:
    err = traceback.format_exc(limit=4)
print(json.dumps({"stdout": buf.getvalue(), "error": err}))
'''

MAX_OUT = 6000


def run_python(code: str, df, timeout: int = 20) -> dict:
    """-> {"stdout", "error", "charts": [paths]}"""
    with tempfile.TemporaryDirectory() as tmp:
        payload = json.dumps({"code": code, "df": df.to_json(orient="split")})
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER], input=payload, capture_output=True, encoding="utf-8", errors="replace",
                               timeout=timeout, cwd=tmp, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return {"stdout": "", "error": f"Timed out after {timeout}s", "charts": []}
        out_lines = p.stdout.strip().splitlines()
        try:
            res = json.loads(out_lines[-1])
        except (IndexError, json.JSONDecodeError):
            return {"stdout": "", "error": (p.stderr or p.stdout or "sandbox crashed")[-1500:], "charts": []}
        charts = []
        pngs = sorted(Path(tmp).glob("*.png"))
        if pngs:
            keep = Path(tempfile.gettempdir()) / "inventory_agent_charts"
            keep.mkdir(exist_ok=True)
            for f in pngs:
                dest = keep / f"{abs(hash((code, f.name))) % 10**8}_{f.name}"
                dest.write_bytes(f.read_bytes())
                charts.append(str(dest))
        stdout = res["stdout"]
        if len(stdout) > MAX_OUT:
            stdout = stdout[:MAX_OUT] + f"\n... [truncated, {len(res['stdout'])} chars total]"
        return {"stdout": stdout, "error": res["error"], "charts": charts}
