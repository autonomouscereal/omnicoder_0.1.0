from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import shutil
import subprocess
import sys
import time


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, sha256: Optional[str] = None) -> int:
    _ensure_dir(dest)
    # Prefer aria2c if available (fast, resumable), else curl/wget fallback
    if shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "-x", "8",
            "-s", "8",
            "-c",  # continue
            "--auto-file-renaming=false",
            "-o", dest.name,
            "-d", str(dest.parent),
        ]
        if sha256:
            cmd += ["--checksum", f"sha-256={sha256}"]
        cmd += [url]
        return subprocess.call(cmd)
    if shutil.which("curl"):
        # Try to resume (-C -). If server doesn't support, curl starts over.
        return subprocess.call(["curl", "-L", "-C", "-", "-o", str(dest), url])
    if shutil.which("wget"):
        return subprocess.call(["wget", "-c", "-O", str(dest), url])
    raise RuntimeError("No downloader available (aria2c/curl/wget)")


def _extract(archive: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    name = archive.name.lower()
    print(f"[extract] start: {archive} -> {out_dir}")
    sys.stdout.flush()
    inprog = archive.with_suffix(archive.suffix + ".extracting")
    try:
        inprog.write_text(str(time.time()), encoding='utf-8')
    except Exception:
        pass
    try:
        if name.endswith(".zip"):
            if shutil.which("unzip"):
                # overwrite existing quietly? Keep verbose for progress
                subprocess.check_call(["unzip", "-o", str(archive), "-d", str(out_dir)])
            else:
                # Python fallback with periodic progress prints
                import zipfile
                with zipfile.ZipFile(str(archive), 'r') as zf:
                    infos = zf.infolist()
                    total = len(infos)
                    last_log = time.time()
                    for i, zi in enumerate(infos, 1):
                        zf.extract(zi, str(out_dir))
                        if (i % 500 == 0) or (time.time() - last_log > 5):
                            print(f"[extract] {archive.name}: {i}/{total}")
                            sys.stdout.flush()
                            last_log = time.time()
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            # verbose to show filenames; faster native tar
            subprocess.check_call(["tar", "-xzvf", str(archive), "-C", str(out_dir)])
        elif name.endswith(".tar.bz2") or name.endswith(".tbz2"):
            subprocess.check_call(["tar", "-xjvf", str(archive), "-C", str(out_dir)])
        else:
            print(f"[extract] skip (unknown format): {archive}")
            return
        print(f"[extract] done: {archive}")
    finally:
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass


def _sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch non-HF external datasets listed in profiles/datasets.json:external_sources")
    ap.add_argument("--profiles", type=str, default="profiles/datasets.json")
    ap.add_argument("--keys", type=str, default="", help="Optional space-separated keys to fetch; fetch all when empty")
    args = ap.parse_args()

    cfg = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    sources: Dict[str, dict] = cfg.get("external_sources", {})  # type: ignore
    want = [k.strip() for k in str(args.keys).split() if k.strip()] if args.keys else list(sources.keys())
    out_summary = {}
    for key in want:
        spec = sources.get(key)
        if not isinstance(spec, dict):
            continue
        url = str(spec.get("url", "")).strip()
        dest = Path(str(spec.get("dest", "")).strip())
        extr = bool(spec.get("extract", False))
        sha256 = str(spec.get("sha256", "")).strip() or None
        size_bytes = int(spec.get("size_bytes", 0)) if str(spec.get("size_bytes", "")).strip().isdigit() else 0
        if not url or not dest:
            continue
        # Skip/redownload logic
        if dest.exists() and dest.is_file():
            try:
                if sha256:
                    cur = _sha256_file(dest)
                    if cur.lower() == sha256.lower():
                        print(f"[skip] {key}: already present and sha256 matches")
                        out_summary[key] = str(dest)
                        # Optionally mark extraction done if marker exists
                        pass
                        if extr:
                            marker = dest.with_suffix(dest.suffix + ".extract.ok")
                            if not marker.exists():
                                _extract(dest, dest.parent)
                                marker.write_text(cur, encoding='utf-8')
                        continue
                elif size_bytes > 0 and dest.stat().st_size >= size_bytes:
                    print(f"[skip] {key}: file present and size >= expected")
                    out_summary[key] = str(dest)
                    if extr:
                        marker = dest.with_suffix(dest.suffix + ".extract.ok")
                        if not marker.exists():
                            _extract(dest, dest.parent)
                            marker.write_text(str(dest.stat().st_size), encoding='utf-8')
                    continue
            except Exception:
                pass
        print(f"[fetch] {key}: {url} -> {dest}")
        rc = _download(url, dest, sha256=sha256)
        if rc != 0:
            print(f"[warn] download failed for {key} (rc={rc})")
            continue
        # Verify after download
        if sha256:
            try:
                cur = _sha256_file(dest)
                if cur.lower() != sha256.lower():
                    print(f"[warn] sha256 mismatch for {key}: got {cur}")
            except Exception as e:
                print(f"[warn] sha256 compute failed for {key}: {e}")
        if extr:
            try:
                # Extract alongside destination path
                out_dir = dest.parent
                marker = dest.with_suffix(dest.suffix + ".extract.ok")
                if not marker.exists():
                    _extract(dest, out_dir)
                    # record marker (hash or size)
                    try:
                        val = _sha256_file(dest) if sha256 is None else sha256
                    except Exception:
                        val = str(dest.stat().st_size)
                    marker.write_text(val, encoding='utf-8')
            except Exception as e:
                print(f"[warn] extract failed for {key}: {e}")
        out_summary[key] = str(dest)
    Path("weights").mkdir(parents=True, exist_ok=True)
    Path("weights/external_fetch.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()


