from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional


class ToolRegistry:
    """
    Minimal tool-use protocol skeleton. Parses inline tool call tags in text and
    dispatches to registered Python callables. Designed to be extended to external
    tools/APIs and treated as another modality in the system.

    Syntax (inline in generated text):
      <tool:name {json_args}>

    Example:
      "What is 2+2? <tool:calculator {\"expr\": \"2+2\"}>"

    The registry returns a dict with replacements that callers can insert into
    the context. On device, this can be used to implement offline tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(self, name: str, func: Callable[[Dict[str, Any]], Any]) -> None:
        self._tools[name.strip().lower()] = func

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._tools

    def invoke(self, name: str, args: Dict[str, Any]) -> Any:
        fn = self._tools.get(name.strip().lower())
        if fn is None:
            raise KeyError(f"Unknown tool: {name}")
        return fn(args)

    # Convenience alias to mirror other registries; avoids confusion in callers
    def call(self, name: str, args: Dict[str, Any]) -> Any:  # noqa: D401
        """Invoke a tool by name with args. Alias to invoke()."""
        return self.invoke(name, args)

    def parse_and_invoke_all(self, text: str) -> Dict[str, Any]:
        """
        Find all tool tags in text, invoke them, and return a mapping from the
        original tag string to the tool output (JSON-serializable preferred).
        """
        results: Dict[str, Any] = {}
        for full, name, argstr in _find_tool_tags(text):
            try:
                args = json.loads(argstr) if argstr.strip() else {}
            except Exception:
                args = {}
            try:
                out = self.invoke(name, args)
            except Exception as e:
                out = {"error": str(e)}
            results[full] = out
        return results


_TOOL_PATTERN = re.compile(r"<tool:([a-zA-Z0-9_\-]+)\s*(\{.*?\})?>")


def _find_tool_tags(text: str) -> list[tuple[str, str, str]]:
    out = []
    for m in _TOOL_PATTERN.finditer(text or ""):
        full = m.group(0)
        name = m.group(1) or ""
        argstr = m.group(2) or "{}"
        out.append((full, name, argstr))
    return out


def default_calculator(args: Dict[str, Any]) -> Any:
    expr = str(args.get("expr", "")).strip()
    if not expr:
        return {"error": "empty expr"}
    try:
        # Safe eval subset: only numbers and operators
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return {"error": "invalid expr"}
        val = eval(expr, {"__builtins__": {}}, {})  # type: ignore
        return {"result": float(val)}
    except Exception as e:
        return {"error": str(e)}


def _local_search(args: Dict[str, Any]) -> Any:
    # Very small offline stub: looks in a provided corpus dict for substring matches
    query = str(args.get("q", "")).strip().lower()
    corpus = args.get("corpus", {})
    if not isinstance(corpus, dict) or not query:
        return {"results": []}
    hits = []
    for k, v in corpus.items():
        try:
            if query and (query in str(v).lower() or query in str(k).lower()):
                hits.append({"key": k, "text": v})
        except Exception:
            continue
    return {"results": hits[:5]}


def _local_db_get(args: Dict[str, Any]) -> Any:
    # Simple key-value fetch from a provided dict
    db = args.get("db", {})
    key = args.get("key", None)
    if isinstance(db, dict) and key in db:
        return {"value": db[key]}
    return {"error": "not found"}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("calculator", default_calculator)
    reg.register("search", _local_search)
    reg.register("db_get", _local_db_get)
    # Convenience: default KB using env root/topk
    import os as _os
    _kb_root = _os.getenv("OMNICODER_KB_ROOT", "").strip()
    _kb_topk = int(_os.getenv("OMNICODER_KB_TOPK", "5"))
    if _kb_root:
        def _kb(args: Dict[str, Any]) -> Any:
            q = str(args.get("q", ""))
            if not q:
                return {"results": []}
            return _kb_search({"root": _kb_root, "q": q, "top_k": _kb_topk})
        reg.register("kb", _kb)
    # Lightweight local KB tools (file/directory backed)
    def _kb_search(args: Dict[str, Any]) -> Any:
        import os as _os
        import json as _json
        root = str(args.get("root", "")).strip()
        query = str(args.get("q", "")).strip().lower()
        top_k = int(args.get("top_k", 5))
        if not root or not query:
            return {"results": []}
        paths: list[str] = []
        try:
            if _os.path.isdir(root):
                for dirpath, _dirnames, filenames in _os.walk(root):
                    # Skip hidden directories
                    parts = {p.lower() for p in dirpath.split(_os.sep)}
                    if any(seg.startswith('.') for seg in parts):
                        continue
                    for fn in filenames:
                        if any(fn.lower().endswith(ext) for ext in (".txt", ".md", ".jsonl")):
                            paths.append(_os.path.join(dirpath, fn))
            elif _os.path.isfile(root):
                paths.append(root)
        except Exception:
            paths = []
        results = []
        for p in paths:
            try:
                if p.lower().endswith('.jsonl'):
                    with open(p, 'r', encoding='utf-8') as f:
                        for ln, line in enumerate(f, start=1):
                            s = line.strip()
                            if not s:
                                continue
                            try:
                                obj = _json.loads(s)
                                text = str(obj.get('text', obj))
                            except Exception:
                                text = s
                            low = text.lower()
                            if query in low:
                                pos = low.find(query)
                                start = max(0, pos - 80)
                                end = min(len(text), pos + 80)
                                results.append({"path": p, "line": ln, "score": 1.0, "snippet": text[start:end]})
                else:
                    text = open(p, 'r', encoding='utf-8', errors='ignore').read()
                    low = text.lower()
                    if query in low:
                        pos = low.find(query)
                        start = max(0, pos - 80)
                        end = min(len(text), pos + 80)
                        results.append({"path": p, "score": 1.0, "snippet": text[start:end]})
            except Exception:
                continue
        return {"results": results[: max(1, top_k)]}
    reg.register("kb_search", _kb_search)

    def _kb_get(args: Dict[str, Any]) -> Any:
        import os as _os
        import json as _json
        path = str(args.get("path", "")).strip()
        if not path:
            return {"error": "path required"}
        if not _os.path.exists(path):
            return {"error": "not found"}
        try:
            if path.lower().endswith('.jsonl'):
                out = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            out.append(_json.loads(s))
                        except Exception:
                            out.append({"text": s})
                return {"records": out}
            else:
                txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
                return {"text": txt}
        except Exception as e:
            return {"error": str(e)}
    reg.register("kb_get", _kb_get)
    # NEW: Safe local Python execution (restricted) for code-based self-check.
    # This executes pure-Python numerics and prints, captures stdout, returns result.
    # Security: no builtins, no imports; only a restricted dict is provided.
    def _exec_py(args: Dict[str, Any]) -> Any:
        src = str(args.get("code", ""))
        if not src.strip():
            return {"error": "empty code"}
        try:
            import io as _io
            import contextlib as _ctx
            buf = _io.StringIO()
            loc: Dict[str, Any] = {}
            glb: Dict[str, Any] = {"__builtins__": {}}
            with _ctx.redirect_stdout(buf):
                exec(src, glb, loc)
            out = buf.getvalue()
            return {"stdout": out, "locals": {k: str(v)[:256] for k, v in loc.items()}}
        except Exception as e:
            return {"error": str(e)}
    reg.register("py_exec", _exec_py)
    # Hook to code verifier tool (shells out to our CLI when requested)
    def _code_verify(args: Dict[str, Any]) -> Any:
        try:
            code = str(args.get("code", ""))
            tests = str(args.get("tests", ""))
            out_json = str(args.get("out", "weights/code_verify_result.json"))
            if not tests:
                return {"error": "tests path required"}
            import subprocess, sys as _sys
            cmd = [_sys.executable, "-m", "omnicoder.tools.code_verifier", "--tests_file", tests, "--out_json", out_json]
            if code:
                cmd += ["--code_text", code]
            rc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return {"ok": (rc.returncode == 0), "stdout": rc.stdout[-2000:], "stderr": rc.stderr[-2000:], "out": out_json}
        except Exception as e:
            return {"error": str(e)}
    reg.register("code_verify", _code_verify)
    # Symbolic solver (best-effort via sympy; falls back to error)
    def _solve(args: Dict[str, Any]) -> Any:
        expr = str(args.get("expr", "")).strip()
        var = str(args.get("var", "x")).strip() or "x"
        if not expr:
            return {"error": "empty expr"}
        try:
            import sympy as sp  # type: ignore
            x = sp.symbols(var)
            sol = sp.solve(sp.Eq(sp.sympify(expr.split('=')[0]), sp.sympify(expr.split('=')[1]) if '=' in expr else 0), x)
            return {"solution": [str(s) for s in sol]}
        except Exception as e:
            return {"error": f"sympy unavailable or failed: {e}"}
    reg.register("solve", _solve)
    return reg


