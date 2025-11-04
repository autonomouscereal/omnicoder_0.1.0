from __future__ import annotations

"""
autofetch_datasets: Download small but useful public datasets for multiple modalities
and materialize them under data/ so training can run on real corpora out-of-the-box.

Requires: pip install datasets

Outputs (defaults under /workspace):
- data/text/wikitext2.txt
- data/code/mbpp.jsonl
- data/vl/images/*.jpg and data/vl.jsonl (COCO captions subset)
- data/asr/wavs/*.wav and data/asr/transcripts.jsonl (LibriSpeech dev-clean subset)

Use environment variables or CLI args to cap per‑modality item counts.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    try:
        print(f"[autofetch] {msg}", flush=True)
    except Exception:
        pass
def _aria2c(url: str, out: Path) -> bool:
    """Download with resume support; returns True if file exists after run."""
    try:
        import subprocess, shlex
        _ensure_dir(out.parent)
        if out.exists() and out.stat().st_size > 0:
            _log(f"skip download (exists): {out}")
            return True
        _log(f"downloading: {url} -> {out}")
        # Prefer aria2c; fallback to curl or wget if missing
        if shutil.which('aria2c'):
            cmd = f"aria2c -c -x 8 -s 8 -k 4M -o {shlex.quote(out.name)} --dir {shlex.quote(str(out.parent))} {shlex.quote(url)}"
        elif shutil.which('curl'):
            cmd = f"curl -L --fail --retry 5 --retry-delay 2 -o {shlex.quote(str(out))} {shlex.quote(url)}"
        elif shutil.which('wget'):
            cmd = f"wget -O {shlex.quote(str(out))} {shlex.quote(url)}"
        else:
            raise RuntimeError('no aria2c/curl/wget available')
        rc = subprocess.call(cmd, shell=True)
        ok = rc == 0 and out.exists() and out.stat().st_size > 0
        _log(f"download {'ok' if ok else 'failed'}: {out}")
        return ok
    except Exception as e:
        _log(f"download error: {url} -> {out}: {e}")
        # Fallback: try Python requests streaming when aria2c is unavailable
        try:
            import requests  # type: ignore
            _ensure_dir(out.parent)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            ok = out.exists() and out.stat().st_size > 0
            _log(f"download via requests {'ok' if ok else 'failed'}: {out}")
            return ok
        except Exception as e2:
            _log(f"download fallback error: {url} -> {out}: {e2}")
            return False


def _count_lines(p: Path) -> int:
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _append_lines_until(path: Path, texts: Iterable[str], target_lines: int) -> int:
    """Append newline-terminated strings until file has target_lines; returns appended count."""
    _ensure_dir(path.parent)
    existing = _count_lines(path)
    if existing >= target_lines:
        return 0
    to_add = target_lines - existing
    appended = 0
    with open(path, 'a', encoding='utf-8') as f:
        for s in texts:
            f.write(s.rstrip('\n') + '\n')
            appended += 1
            if appended >= to_add:
                break
    return appended


def fetch_external_archives() -> dict[str, str]:
    """Fetch large, open archives from non-HF mirrors (tokenless) when available.

    Sources (examples; mirrors may change):
    - Code: CodeParrot dumps, permissive Git mirrors (compressed JSONL)
    - Vision: CC12M/CC3M-like mirrors (captions+urls with downloader)
    - Video: UCF101/Kinetics mirrors when available
    """
    out: dict[str, str] = {}
    # Example: codeparrot URL mirror (if present)
    try:
        url = os.getenv("OMNICODER_CODE_MIRROR", "")
        if url:
            dest = Path("data/code/mirror_codeparrot.jsonl.zst")
            if _aria2c(url, dest):
                out["code_mirror_zst"] = str(dest)
    except Exception:
        pass
    # Example: captions TSV mirror (LAION-like) for VL
    try:
        url2 = os.getenv("OMNICODER_VL_MIRROR", "")
        if url2:
            dest2 = Path("data/vl/mirror_captions.tsv")
            if _aria2c(url2, dest2):
                out["vl_mirror_tsv"] = str(dest2)
    except Exception:
        pass
    # Example: video tar shard
    try:
        url3 = os.getenv("OMNICODER_VIDEO_MIRROR", "")
        if url3:
            dest3 = Path("data/video/videos_000.tar")
            if _aria2c(url3, dest3):
                out["video_mirror_tar"] = str(dest3)
    except Exception:
        pass
    return out


def fetch_coco_official(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        root = Path("data/coco2017")
        _ensure_dir(root)
        imgs_zip = root / "train2017.zip"
        ann_zip = root / "annotations_trainval2017.zip"
        # Official COCO 2017 URLs
        url_imgs = "http://images.cocodataset.org/zips/train2017.zip"
        url_ann = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        _aria2c(url_imgs, imgs_zip)
        _aria2c(url_ann, ann_zip)
        # Extract minimal needed
        import zipfile, json
        imgs_dir = root / "train2017"
        ann_dir = root / "annotations"
        if not imgs_dir.exists() and imgs_zip.exists():
            _log("extracting COCO train2017.zip ...")
            with zipfile.ZipFile(str(imgs_zip), 'r') as z:
                z.extractall(str(root))
        if not ann_dir.exists() and ann_zip.exists():
            _log("extracting COCO annotations_trainval2017.zip ...")
            with zipfile.ZipFile(str(ann_zip), 'r') as z:
                z.extractall(str(root))
        # Build a small VL jsonl from captions
        cap_json = ann_dir / "captions_train2017.json"
        vl_out = Path("data/vl_coco.jsonl")
        if cap_json.exists() and imgs_dir.exists():
            caps = json.loads(cap_json.read_text(encoding='utf-8'))
            id_to_file = {int(img.get('id')): str((imgs_dir / img.get('file_name', '')).as_posix()) for img in caps.get('images', [])}
            existing = _count_lines(vl_out)
            target = max(1, limit)
            if existing < target:
                _log(f"building COCO VL jsonl: {existing} -> {target} lines")
                import json as _json
                n = 0
                with open(vl_out, 'a', encoding='utf-8') as f:
                    # Skip first `existing` items deterministically
                    skipped = 0
                    for ann in caps.get('annotations', []):
                        image_id = int(ann.get('image_id', -1))
                        txt = str(ann.get('caption', ''))
                        imgp = id_to_file.get(image_id, '')
                        if imgp and txt.strip():
                            if skipped < existing:
                                skipped += 1
                                continue
                            f.write(_json.dumps({"image": imgp, "text": txt}, ensure_ascii=False) + "\n")
                            n += 1
                            if existing + n >= target:
                                break
            out["vl_coco_jsonl"] = str(vl_out)
            out["coco_images_dir"] = str(imgs_dir)
    except Exception as e:
        out["coco_error"] = str(e)
    return out


def fetch_librispeech_official(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        root = Path("data/librispeech")
        _ensure_dir(root)
        url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        tar = root / "dev-clean.tar.gz"
        if not tar.exists():
            _aria2c(url, tar)
        if tar.exists():
            import tarfile
            with tarfile.open(str(tar), 'r:gz') as tf:
                tf.extractall(str(root))
        wav_root = next((p for p in root.glob('LibriSpeech/dev-clean') if p.exists()), root / 'LibriSpeech/dev-clean')
        # Build transcripts jsonl
        tr_jsonl = Path("data/asr/transcripts.jsonl")
        _ensure_dir(tr_jsonl.parent)
        import json
        n = 0
        with open(tr_jsonl, 'w', encoding='utf-8') as f:
            for txtf in wav_root.rglob('*.trans.txt'):
                try:
                    lines = txtf.read_text(encoding='utf-8', errors='ignore').splitlines()
                    for line in lines:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            utt, text = parts
                            wavp = (txtf.parent / (utt + '.flac'))
                            if wavp.exists():
                                f.write(json.dumps({"path": str(wavp.as_posix()), "text": text}, ensure_ascii=False) + "\n")
                                n += 1
                                if n >= limit:
                                    raise StopIteration
                except StopIteration:
                    break
                except Exception:
                    continue
        out["asr_dir"] = str(Path("data/asr").as_posix())
    except Exception as e:
        out["librispeech_error"] = str(e)
    return out


def fetch_ljspeech_official(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        root = Path("data/ljspeech")
        _ensure_dir(root)
        url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        tbz = root / "LJSpeech-1.1.tar.bz2"
        if not tbz.exists():
            _aria2c(url, tbz)
        if tbz.exists():
            import tarfile
            with tarfile.open(str(tbz), 'r:bz2') as tf:
                tf.extractall(str(root))
        meta = next((p for p in root.glob('LJSpeech-1.1/metadata.csv') if p.exists()), None)
        wavs = next((p for p in root.glob('LJSpeech-1.1/wavs') if p.exists()), None)
        if meta and wavs:
            lines = meta.read_text(encoding='utf-8', errors='ignore').splitlines()
            texts = []
            for i, line in enumerate(lines[:max(1, limit)]):
                try:
                    fid, _, text = line.split('|', 2)
                    texts.append(text)
                except Exception:
                    continue
            tts_dir = Path('data/tts'); _ensure_dir(tts_dir)
            (tts_dir / 'texts.txt').write_text("\n".join(texts), encoding='utf-8')
            out["tts_texts"] = str((tts_dir / 'texts.txt').as_posix())
    except Exception as e:
        out["ljspeech_error"] = str(e)
    return out


def fetch_wit_official(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        root = Path('data/wit'); _ensure_dir(root)
        # First shard of train TSV
        url = "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00000-of-00010.tsv.gz"
        gz = root / 'wit_v1.train.all-00000-of-00010.tsv.gz'
        if not gz.exists():
            _aria2c(url, gz)
        if gz.exists():
            import gzip
            tsv = root / 'wit_v1.train.part.tsv'
            with gzip.open(gz, 'rb') as src, open(tsv, 'wb') as dst:
                dst.write(src.read())
            out["wit_tsv"] = str(tsv)
    except Exception as e:
        out["wit_error"] = str(e)
    return out


def _download_tsv_env(env_key: str, default_name: str, out_dir: Path) -> str | None:
    url = os.getenv(env_key, "").strip()
    if not url:
        return None
    _ensure_dir(out_dir)
    dest = out_dir / default_name
    if _aria2c(url, dest):
        return str(dest)
    return None


def fetch_cc12m_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        p = _download_tsv_env("OMNICODER_CC12M_TSV", "cc12m.tsv.gz", Path("data/vl/cc12m"))
        if p:
            out["cc12m_tsv"] = p
    except Exception:
        pass
    return out


def fetch_coyo_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        p = _download_tsv_env("OMNICODER_COYO_TSV", "coyo.tsv", Path("data/vl/coyo"))
        if p:
            out["coyo_tsv"] = p
    except Exception:
        pass
    return out


def fetch_laion_shards() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        list_url = os.getenv("OMNICODER_LAION_SHARDS_LIST", "").strip()
        max_n = int(os.getenv("OMNICODER_LAION_SHARDS_MAX", "4"))
        if not list_url:
            return out
        lists_dir = Path("data/vl/laion"); _ensure_dir(lists_dir)
        list_path = lists_dir / "shards.txt"
        if _aria2c(list_url, list_path):
            out["laion_shards_list"] = str(list_path)
            shards_dir = lists_dir / "shards"; _ensure_dir(shards_dir)
            # Download first N shards
            try:
                urls = [u.strip() for u in list_path.read_text(encoding='utf-8', errors='ignore').splitlines() if u.strip()]
                import itertools
                for url in itertools.islice(urls, max(1, max_n)):
                    name = url.split('/')[-1] or f"shard_{len(list(shards_dir.glob('*'))):05d}"
                    _aria2c(url, shards_dir / name)
                out["laion_shards_dir"] = str(shards_dir)
            except Exception:
                pass
    except Exception:
        pass
    return out



def fetch_text(limit: int) -> str:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{max(1,limit)}]")
    outp = Path("data/text/wikitext2.txt")
    _ensure_dir(outp.parent)
    existing = _count_lines(outp)
    if existing >= limit:
        _log(f"skip wikitext2: have {existing} lines >= {limit}")
        return str(outp)
    _log(f"writing wikitext2: {existing} -> {limit}")
    appended = 0
    with open(outp, "a", encoding="utf-8") as f:
        skipped = 0
        for ex in ds:
            if skipped < existing:
                skipped += 1
                continue
            txt = str(ex.get("text", ""))
            if txt.strip():
                f.write(txt.rstrip("\n") + "\n")
                appended += 1
                if existing + appended >= limit:
                    break
    return str(outp)


def fetch_text_more(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    # FineWeb-Edu (high-quality educational web): open and streamable
    try:
        from datasets import load_dataset  # type: ignore
        ds_stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        outp = Path("data/text/fineweb_edu.txt"); _ensure_dir(outp.parent)
        existing = _count_lines(outp)
        if existing < limit:
            _log(f"append FineWeb-Edu: {existing} -> {limit}")
            def _iter() -> Iterable[str]:
                n = 0
                for ex in ds_stream:
                    txt = str(ex.get("text", ""))
                    if txt.strip():
                        yield txt
                        n += 1
                        if existing + n >= limit:
                            break
            _append_lines_until(outp, _iter(), limit)
        out["fineweb_edu"] = str(outp)
    except Exception as e:
        out["fineweb_edu_error"] = str(e)

    # OpenWebText (HF streaming)
    try:
        from datasets import load_dataset  # type: ignore
        ds_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        outp = Path("data/text/openwebtext.txt"); _ensure_dir(outp.parent)
        existing = _count_lines(outp)
        if existing < limit:
            _log(f"append OpenWebText: {existing} -> {limit}")
            def _iter2() -> Iterable[str]:
                n = 0
                for ex in ds_stream:
                    txt = str(ex.get("text", ""))
                    if txt.strip():
                        yield txt
                        n += 1
                        if existing + n >= limit:
                            break
            _append_lines_until(outp, _iter2(), limit)
        out["openwebtext"] = str(outp)
    except Exception as e:
        out["openwebtext_error"] = str(e)
    try:
        from datasets import load_dataset  # type: ignore
        ds_stream = load_dataset("c4", "en", split="train", streaming=True)
        outp = Path("data/text/c4_en.txt"); _ensure_dir(outp.parent)
        existing = _count_lines(outp)
        if existing < limit:
            _log(f"append C4-en: {existing} -> {limit}")
            def _iter3() -> Iterable[str]:
                n = 0
                for ex in ds_stream:
                    txt = str(ex.get("text", ""))
                    if txt.strip():
                        yield txt
                        n += 1
                        if existing + n >= limit:
                            break
            _append_lines_until(outp, _iter3(), limit)
        out["c4_en"] = str(outp)
    except Exception as e:
        out["c4_en_error"] = str(e)
    try:
        from datasets import load_dataset  # type: ignore
        ds_stream = load_dataset("bookcorpusopen", split="train", streaming=True)
        outp = Path("data/text/bookcorpusopen.txt"); _ensure_dir(outp.parent)
        existing = _count_lines(outp)
        if existing < limit:
            _log(f"append BookCorpusOpen: {existing} -> {limit}")
            def _iter4() -> Iterable[str]:
                n = 0
                for ex in ds_stream:
                    txt = str(ex.get("text", ""))
                    if txt.strip():
                        yield txt
                        n += 1
                        if existing + n >= limit:
                            break
            _append_lines_until(outp, _iter4(), limit)
        out["bookcorpusopen"] = str(outp)
    except Exception as e:
        out["bookcorpusopen_error"] = str(e)
    return out


def fetch_code(limit: int) -> str:
    from datasets import load_dataset  # type: ignore
    # MBPP: programming problems with solutions (good for code distill/eval)
    ds = load_dataset("mbpp", "full", split=f"train[:{max(1,limit)}]")
    outp = Path("data/code/mbpp.jsonl")
    _ensure_dir(outp.parent)
    import json
    with open(outp, "w", encoding="utf-8") as f:
        for ex in ds:
            # Normalize to {prompt, solution}
            prompt = str(ex.get("text", "")) or str(ex.get("prompt", ""))
            solution = str(ex.get("code", "")) or str(ex.get("solution", ""))
            f.write(json.dumps({"prompt": prompt, "solution": solution}, ensure_ascii=False) + "\n")
    return str(outp)


def fetch_code_more(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        # CodeParrot Clean (permissive code dataset)
        ds_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        outp = Path("data/code/codeparrot_clean.jsonl"); _ensure_dir(outp.parent)
        import json
        existing = _count_lines(outp)
        if existing < limit:
            _log(f"append CodeParrot-Clean: {existing} -> {limit}")
            with open(outp, "a", encoding="utf-8") as f:
                n = 0
                for ex in ds_stream:
                    content = str(ex.get("content", "")) or str(ex.get("code", ""))
                    path = str(ex.get("repo_name", ""))
                    if content.strip():
                        if n < existing:
                            n += 1
                            continue
                        f.write(json.dumps({"prompt": path, "solution": content}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
        out["codeparrot_clean"] = str(outp)
    except Exception as e:
        out["codeparrot_clean_error"] = str(e)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("code_x_glue_ct_code_to_text", "ruby", split=f"train[:{max(1,limit)}]")
        outp = Path("data/code/code_to_text_ruby.jsonl"); _ensure_dir(outp.parent)
        import json
        with open(outp, "w", encoding="utf-8") as f:
            for ex in ds:
                code = str(ex.get("code", "")); doc = str(ex.get("docstring", ""))
                f.write(json.dumps({"prompt": code, "solution": doc}, ensure_ascii=False) + "\n")
        out["code_to_text_ruby"] = str(outp)
    except Exception as e:
        out["code_to_text_ruby_error"] = str(e)
    return out


def fetch_vl(limit: int) -> dict[str,str]:
    from datasets import load_dataset  # type: ignore
    # Use COCO captions validation subset; write images and a captions jsonl
    # Use laion-coco (open captions) as a fallback when coco_captions is unavailable
    try:
        subset = f"validation[:{max(1,limit)}]"
        ds = load_dataset("coco_captions", "2017", split=subset)
    except Exception:
        ds = load_dataset("laion/laion-coco", split=f"train[:{max(1,limit)}]")
    img_dir = Path("data/vl/images")
    _ensure_dir(img_dir)
    jsonl = Path("data/vl.jsonl")
    import json
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            try:
                img = ex["image"]  # PIL.Image
                caps = ex.get("captions", []) or []
                txt = str(caps[0]["caption"]) if caps and isinstance(caps[0], (list, tuple, dict)) else str(caps[0]) if caps else ""
                name = f"coco_{i:05d}.jpg"
                out = img_dir / name
                img.save(out)
                f.write(json.dumps({"image": str(out.as_posix()), "text": txt}, ensure_ascii=False) + "\n")
            except Exception:
                continue
    return {"images_dir": str(img_dir), "vl_jsonl": str(jsonl)}


def fetch_vl_more(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        # SBU captions (URL-based)
        ds = load_dataset("sbu_captions", split=f"train[:{max(1,limit)}]")
        img_dir = Path("data/vl_cc/images"); _ensure_dir(img_dir)
        jsonl = Path("data/vl_cc.jsonl")
        import json, requests
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    url = str(ex.get("url", "") or ex.get("image_url", "")); cap = str(ex.get("caption", ""))
                    if not url:
                        continue
                    name = img_dir / f"cc_{i:05d}.jpg"
                    r = requests.get(url, timeout=5)
                    if r.ok:
                        with open(name, "wb") as g:
                            g.write(r.content)
                        f.write(json.dumps({"image": str(name.as_posix()), "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["vl_cc_images_dir"] = str(img_dir)
        out["vl_cc_jsonl"] = str(jsonl)
    except Exception as e:
        out["vl_cc_error"] = str(e)
    # LAION-400M TSV shard (official open links may be provided via env)
    try:
        laion_url = os.getenv("OMNICODER_LAION400M_TSV", "")
        if laion_url:
            laion_tsv = Path("data/vl/laion400m.tsv")
            if _aria2c(laion_url, laion_tsv):
                out["laion400m_tsv"] = str(laion_tsv)
    except Exception:
        pass
    return out


def fetch_vqa(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        # Try VQAv2 first; fall back to TextCaps
        try:
            ds = load_dataset("vizwiz", name="2020_vqa_main", split=f"train[:{max(1,limit)}]")
        except Exception:
            try:
                ds = load_dataset("visualqa/vqav2", split=f"train[:{max(1,limit)}]")
            except Exception:
                ds = load_dataset("TextCaps", split=f"train[:{max(1,limit)}]")
        img_dir = Path("data/vqa/images"); _ensure_dir(img_dir)
        jsonl = Path("data/vqa.jsonl")
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    q = str(ex.get("question", ""))
                    ans = ex.get("answers") or ex.get("answer")
                    a = ""
                    if isinstance(ans, list) and ans:
                        a = str(ans[0])
                    elif isinstance(ans, str):
                        a = ans
                    name = img_dir / f"vqa_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["vqa_images_dir"] = str(img_dir)
        out["vqa_jsonl"] = str(jsonl)
    except Exception as e:
        out["vqa_error"] = str(e)
    return out


def fetch_refcoco_small(limit: int) -> dict[str, str]:
    """Fetch a tiny RefCOCO split to bootstrap grounding. Writes data/refcoco.jsonl."""
    out: dict[str, str] = {}
    try:
        env_jsonl = os.getenv("OMNICODER_REFCOCO_JSONL", "").strip()
        if env_jsonl and Path(env_jsonl).exists():
            out["refcoco_jsonl"] = env_jsonl
            return out
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("refcocog", "umd", split=f"validation[:{max(1,limit)}]")
        _ensure_dir(Path("data"))
        jsonl = Path("data/refcoco.jsonl")
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    img_path = ex.get("image_path") or ex.get("image")
                    expr = ex.get("raw") or ex.get("refexp") or ex.get("expression") or ""
                    bbox = ex.get("bbox") or {}
                    if img_path and expr:
                        f.write(json.dumps({"image": str(img_path), "text": str(expr), "bbox": bbox}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["refcoco_jsonl"] = str(jsonl)
    except Exception as e:
        out["refcoco_error"] = str(e)
    return out


def fetch_clotho_small(limit: int) -> dict[str, str]:
    """Fetch a tiny Clotho subset for audio captioning; writes data/clotho.jsonl."""
    out: dict[str, str] = {}
    try:
        env_jsonl = os.getenv("OMNICODER_CLOTHO_JSONL", "").strip()
        if env_jsonl and Path(env_jsonl).exists():
            out["clotho_jsonl"] = env_jsonl
            return out
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("cdminix/clotho_captions", split=f"validation[:{max(1,limit)}]")
        wav_dir = Path("data/asr/wavs"); _ensure_dir(wav_dir)
        jsonl = Path("data/clotho.jsonl")
        import json, soundfile as sf, numpy as np  # type: ignore
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    audio = ex.get("audio", {})
                    wav = audio.get("array"); sr = audio.get("sampling_rate")
                    caps = ex.get("captions", [])
                    cap = str(caps[0]) if isinstance(caps, list) and caps else ""
                    if wav is None or sr is None:
                        continue
                    p = wav_dir / f"clotho_{i:05d}.wav"
                    sf.write(p.as_posix(), np.asarray(wav, dtype=np.float32), int(sr))
                    f.write(json.dumps({"path": p.as_posix(), "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["clotho_jsonl"] = str(jsonl)
        out["clotho_wavs_dir"] = str(wav_dir.as_posix())
    except Exception as e:
        out["clotho_error"] = str(e)
    return out


def fetch_fsd50k_small(limit: int) -> dict[str, str]:
    """Fetch a tiny FSD50K subset; writes data/fsd50k.jsonl with {path,label}."""
    out: dict[str, str] = {}
    try:
        env_jsonl = os.getenv("OMNICODER_FSD50K_JSONL", "").strip()
        if env_jsonl and Path(env_jsonl).exists():
            out["fsd50k_jsonl"] = env_jsonl
            return out
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("ashraq/fsd50k", split=f"train[:{max(1,limit)}]")
        wav_dir = Path("data/asr/wavs"); _ensure_dir(wav_dir)
        jsonl = Path("data/fsd50k.jsonl")
        import json, soundfile as sf, numpy as np  # type: ignore
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    audio = ex.get("audio", {})
                    wav = audio.get("array"); sr = audio.get("sampling_rate")
                    label = ex.get("label") or ex.get("class") or "unknown"
                    if wav is None or sr is None:
                        continue
                    p = wav_dir / f"fsd_{i:05d}.wav"
                    sf.write(p.as_posix(), np.asarray(wav, dtype=np.float32), int(sr))
                    f.write(json.dumps({"path": p.as_posix(), "label": str(label)}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["fsd50k_jsonl"] = str(jsonl)
    except Exception as e:
        out["fsd50k_error"] = str(e)
    return out


def fetch_asr(limit: int) -> dict[str,str]:
    from datasets import load_dataset, Audio  # type: ignore
    # Prefer LibriSpeech; fallback to Common Voice English
    try:
        ds = load_dataset("librispeech_asr", "clean", split=f"validation[:{max(1,limit)}]")
    except Exception:
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split=f"test[:{max(1,limit)}]")
    # Ensure audio is decoded to 16k wav
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    wav_dir = Path("data/asr/wavs")
    _ensure_dir(wav_dir)
    tr_path = Path("data/asr/transcripts.jsonl")
    import json
    with open(tr_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            try:
                arr = ex["audio"]["array"]
                sr = int(ex["audio"]["sampling_rate"])
                import soundfile as sf  # type: ignore
                out = wav_dir / f"ls_{i:05d}.wav"
                sf.write(out, arr, sr)
                text = str(ex.get("text", ""))
                f.write(json.dumps({"path": str(out.as_posix()), "text": text}, ensure_ascii=False) + "\n")
            except Exception:
                continue
    return {"wav_dir": str(wav_dir), "transcripts": str(tr_path)}


def fetch_tts(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("ljspeech", split=f"train[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=22050))
        wav_dir = Path("data/tts/wavs"); _ensure_dir(wav_dir)
        txt_path = Path("data/tts/texts.txt")
        import soundfile as sf  # type: ignore
        texts = []
        for i, ex in enumerate(ds):
            try:
                arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                out = wav_dir / f"ljs_{i:05d}.wav"; sf.write(out, arr, sr)
                texts.append(str(ex.get("text", "")))
            except Exception:
                continue
        if texts:
            txt_path.write_text("\n".join(texts), encoding="utf-8")
        out["tts_wavs_dir"] = str(wav_dir)
        out["tts_texts"] = str(txt_path)
    except Exception as e:
        out["tts_error"] = str(e)
    return out


def fetch_video(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        # Prefer direct mirrors or WebVid CSV if provided
        webvid_csv = os.getenv("OMNICODER_WEBVID_CSV", "")
        if webvid_csv:
            vid_dir = Path("data/video/webvid"); _ensure_dir(vid_dir)
            out["video_manifest_csv"] = webvid_csv
            out["video_dir"] = str(vid_dir)
        else:
            from datasets import load_dataset  # type: ignore
            vid_dir = Path("data/video/frames"); _ensure_dir(vid_dir)
            # Prefer UCF101; fallback to Kinetics when available; else stub
            ds = None
            try:
                ds = load_dataset("ucf101", "ucf101", split=f"train[:{max(1,limit)}]")
            except Exception:
                try:
                    ds = load_dataset("kinetics700", "2020", split=f"train[:{max(1,limit)}]")
                except Exception:
                    ds = None
            if ds is not None:
                import imageio.v2 as iio  # type: ignore
                for i, ex in enumerate(ds):
                    try:
                        v = ex.get("video") or ex.get("video_array")
                        if hasattr(v, "fps") and hasattr(v, "to_numpy"):
                            arr = v.to_numpy()
                            outp = vid_dir / f"clip_{i:05d}.mp4"
                            iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                    except Exception:
                        continue
                out["video_dir"] = str(vid_dir)
    except Exception as e:
        out["video_error"] = str(e)
    return out


def fetch_speech_commands(limit: int) -> dict[str, str]:
    """Fetch a tiny subset of Speech Commands and write WAVs + labels.
    Useful as additional audio wav_dir for audio latent training or ASR smoke.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("speech_commands", "v0.02", split=f"validation[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        wav_dir = Path("data/asr/sc_wavs"); _ensure_dir(wav_dir)
        tr_path = Path("data/asr/sc_transcripts.jsonl")
        import json, soundfile as sf  # type: ignore
        with open(tr_path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                    label = str(ex.get("label", ""))
                    outp = wav_dir / f"sc_{i:05d}.wav"; sf.write(outp, arr, sr)
                    f.write(json.dumps({"path": str(outp.as_posix()), "text": label}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        # Return under wav_dir key too so orchestrator can pick it up
        out["sc_wavs_dir"] = str(wav_dir.as_posix())
        out["wav_dir"] = str(wav_dir.as_posix())
        out["transcripts"] = str(tr_path.as_posix())
    except Exception as e:
        out["speech_commands_error"] = str(e)
    return out


def fetch_textcaps(limit: int) -> dict[str, str]:
    """Fetch a small TextCaps subset and materialize VQA-style JSONL.
    This augments VQA with text-in-the-wild OCR questions.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = None
        try:
            ds = load_dataset("textcaps", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            return out
        img_dir = Path("data/vqa/textcaps"); _ensure_dir(img_dir)
        jsonl = Path("data/vqa/textcaps.jsonl")
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    q = str(ex.get("question", ""))
                    ans = ex.get("answers", [])
                    a = str(ans[0]) if isinstance(ans, list) and ans else ""
                    name = img_dir / f"tc_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["textcaps_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["textcaps_error"] = str(e)
    return out


def build_mm_instructions_small(limit: int) -> dict[str, str]:
    """Synthesize a tiny multimodal instruction dataset by mixing existing VL/VQA JSONLs.
    Outputs: data/mm_instructions.jsonl with {image|video|audio,text,instruction,answer}.
    """
    out: dict[str, str] = {}
    try:
        import json
        tgt = Path("data/mm_instructions.jsonl"); _ensure_dir(tgt.parent)
        # Candidate sources
        sources = [
            Path("data/vl.jsonl"),
            Path("data/vqa.jsonl"),
            Path("data/vl_cc.jsonl"),
            Path("data/audiocaps/captions.jsonl"),
            Path("data/video/vatex_captions.jsonl"),
        ]
        n = 0
        with open(tgt, "w", encoding="utf-8") as f:
            for sp in sources:
                if not sp.exists():
                    continue
                for line in sp.read_text(encoding='utf-8', errors='ignore').splitlines():
                    try:
                        ex = json.loads(line)
                        inst = None
                        ans = None
                        if "question" in ex:
                            inst = ex.get("question", "")
                            ans = ex.get("answer", "")
                        elif "text" in ex:
                            inst = "Describe this input concisely."
                            ans = ex.get("text", "")
                        if inst:
                            ex_out = {k: v for k, v in ex.items() if k in ("image","video","path")}
                            ex_out["instruction"] = inst
                            ex_out["answer"] = ans or ""
                            f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                            n += 1
                            if n >= max(1, int(limit)):
                                raise StopIteration
                    except StopIteration:
                        raise
                    except Exception:
                        continue
        out["mm_instructions_jsonl"] = str(tgt.as_posix())
    except StopIteration:
        out["mm_instructions_jsonl"] = "data/mm_instructions.jsonl"
    except Exception as e:
        out["mm_instructions_error"] = str(e)
    return out


def fetch_llava_instruct_small(limit: int) -> dict[str, str]:
    """Attempt to fetch a tiny LLaVA instruction subset and materialize images+JSONL.
    Tries several dataset ids; degrades gracefully if unavailable.
    Output: data/instruct/llava.jsonl with {image, instruction, answer}
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json, requests  # type: ignore
        from PIL import Image  # type: ignore
        import io
        img_dir = Path("data/instruct/images"); _ensure_dir(img_dir)
        jsonl = Path("data/instruct/llava.jsonl"); _ensure_dir(jsonl.parent)
        ids = [
            "liuhaotian/LLaVA-Instruct-150K",
            "liuhaotian/llava_instruct_150k",
        ]
        ds = None
        for dsid in ids:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["llava_instruct_error"] = "dataset unavailable"
            return out
        n = 0
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    conv = ex.get("conversations") or []
                    if not conv or len(conv) < 2:
                        continue
                    instr = str(conv[0].get("value", ""))
                    ans = str(conv[1].get("value", ""))
                    img_field = ex.get("image") or ex.get("image_url")
                    path = None
                    if hasattr(img_field, "save"):
                        p = img_dir / f"llava_{n:05d}.png"; img_field.save(p); path = p.as_posix()
                    else:
                        if isinstance(img_field, str) and (img_field.startswith("http://") or img_field.startswith("https://")):
                            try:
                                r = requests.get(img_field, timeout=5)
                                if r.ok:
                                    im = Image.open(io.BytesIO(r.content)).convert("RGB")
                                    p = img_dir / f"llava_{n:05d}.png"; im.save(p); path = p.as_posix()
                            except Exception:
                                path = None
                    if path and instr:
                        f.write(json.dumps({"image": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["llava_instruct_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["llava_instruct_error"] = str(e)
    return out


def fetch_sharegpt4v_small(limit: int) -> dict[str, str]:
    """Attempt to fetch ShareGPT4V instructions and materialize a small JSONL.
    Output: data/instruct/sharegpt4v.jsonl
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/instruct/images"); _ensure_dir(img_dir)
        jsonl = Path("data/instruct/sharegpt4v.jsonl")
        ds = None
        for dsid in ["ShareGPT4V/ShareGPT4V-Instruct", "ShareGPT4V/ShareGPT4V", "linjiyuan/ShareGPT4V"]:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["sharegpt4v_error"] = "dataset unavailable"
            return out
        n = 0
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    instr = str(ex.get("instruction", "") or ex.get("question", ""))
                    ans = str(ex.get("answer", "") or ex.get("output", ""))
                    img = ex.get("image") or ex.get("image_pil")
                    path = None
                    if hasattr(img, "save"):
                        p = img_dir / f"sg4v_{n:05d}.png"; img.save(p); path = p.as_posix()
                    if path and instr:
                        f.write(json.dumps({"image": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["sharegpt4v_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["sharegpt4v_error"] = str(e)
    return out


def fetch_minigpt4_instruct_small(limit: int) -> dict[str, str]:
    """Attempt to fetch MiniGPT-4 instruction subset; materialize JSONL.
    Output: data/instruct/minigpt4.jsonl
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/instruct/images"); _ensure_dir(img_dir)
        jsonl = Path("data/instruct/minigpt4.jsonl")
        ds = None
        for dsid in ["MiniGPT4/mini_gpt4_instructions", "akiani/MiniGPT4-Instruction"]:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["minigpt4_error"] = "dataset unavailable"
            return out
        n = 0
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    instr = str(ex.get("instruction", "") or ex.get("question", ""))
                    ans = str(ex.get("answer", "") or ex.get("output", ""))
                    img = ex.get("image") or ex.get("image_pil")
                    path = None
                    if hasattr(img, "save"):
                        p = img_dir / f"mg4_{n:05d}.png"; img.save(p); path = p.as_posix()
                    if path and instr:
                        f.write(json.dumps({"image": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["minigpt4_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["minigpt4_error"] = str(e)
    return out


def fetch_videochat_instruct_small(limit: int) -> dict[str, str]:
    """Attempt to fetch Video-Chat instruction subset; materialize video+instruction JSONL.
    Output: data/instruct/videochat.jsonl
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        import json
        vid_dir = Path("data/instruct/video"); _ensure_dir(vid_dir)
        jsonl = Path("data/instruct/videochat.jsonl")
        ds = None
        for dsid in ["Video-ChatGPT/Video-ChatGPT_instructions", "videochat2/Video-Chat2-Instructions"]:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["videochat_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            n = 0
            for ex in ds:
                try:
                    instr = str(ex.get("instruction", "") or ex.get("question", ""))
                    ans = str(ex.get("answer", "") or ex.get("output", ""))
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"vchat_{n:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and instr:
                        f.write(json.dumps({"video": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["videochat_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["videochat_error"] = str(e)
    return out


def fetch_instructblip_small(limit: int) -> dict[str, str]:
    """Fetch a tiny InstructBLIP/BLIP-2 style instruction subset.
    Output: data/instruct/instructblip.jsonl
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/instruct/images"); _ensure_dir(img_dir)
        jsonl = Path("data/instruct/instructblip.jsonl")
        ds = None
        for dsid in ["Salesforce/instructblip-demo", "Salesforce/blip2-instruct"]:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["instructblip_error"] = "dataset unavailable"
            return out
        n = 0
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    instr = str(ex.get("instruction", "") or ex.get("question", ""))
                    ans = str(ex.get("answer", "") or ex.get("output", ""))
                    img = ex.get("image") or ex.get("image_pil")
                    path = None
                    if hasattr(img, "save"):
                        p = img_dir / f"iblip_{n:05d}.png"; img.save(p); path = p.as_posix()
                    if path and instr:
                        f.write(json.dumps({"image": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["instructblip_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["instructblip_error"] = str(e)
    return out


def fetch_llava_next_small(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/instruct/images"); _ensure_dir(img_dir)
        jsonl = Path("data/instruct/llava_next.jsonl")
        ds = None
        for dsid in ["liuhaotian/llava_next_instruct", "LLaVA-Instruct/llava-next"]:
            try:
                ds = load_dataset(dsid, split=f"train[:{max(1,limit)}]")
                break
            except Exception:
                ds = None
        if ds is None:
            out["llava_next_error"] = "dataset unavailable"
            return out
        n = 0
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    instr = str(ex.get("instruction", "") or ex.get("question", ""))
                    ans = str(ex.get("answer", "") or ex.get("output", ""))
                    img = ex.get("image") or ex.get("image_pil")
                    path = None
                    if hasattr(img, "save"):
                        p = img_dir / f"llnext_{n:05d}.png"; img.save(p); path = p.as_posix()
                    if path and instr:
                        f.write(json.dumps({"image": path, "instruction": instr, "answer": ans}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["llava_next_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["llava_next_error"] = str(e)
    return out


def fetch_activitynet_captions_small(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        import json
        vid_dir = Path("data/video/activitynet"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/activitynet_captions.jsonl")
        ds = None
        try:
            ds = load_dataset("activitynet_captions", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["activitynet_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    v = ex.get("video"); caps = ex.get("captions") or []
                    cap = str(caps[0]) if caps else ""
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"anet_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and cap:
                        f.write(json.dumps({"video": path, "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["activitynet_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["activitynet_error"] = str(e)
    return out


def fetch_charades_sta_small(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json, imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/charades"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/charades_sta.jsonl")
        ds = None
        try:
            ds = load_dataset("charades_sta", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["charades_sta_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    v = ex.get("video"); q = str(ex.get("query", "") or ex.get("sentence", ""))
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"char_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and q:
                        f.write(json.dumps({"video": path, "text": q}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["charades_sta_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["charades_sta_error"] = str(e)
    return out


def fetch_ego4d_nlq_small(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        jsonl = Path("data/video/ego4d_nlq.jsonl")
        ds = None
        try:
            ds = load_dataset("ego4d", "nlq", split=f"validation[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["ego4d_nlq_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = str(ex.get("query", "") or ex.get("question", ""))
                    v = ex.get("video") or ""
                    if v and q:
                        f.write(json.dumps({"video": str(v), "text": q}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["ego4d_nlq_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["ego4d_nlq_error"] = str(e)
    return out


def fetch_pubtables1m_small(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/pubtables/images"); _ensure_dir(img_dir)
        jsonl = Path("data/pubtables/pubtables1m.jsonl")
        ds = None
        try:
            ds = load_dataset("pubtables-1m", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["pubtables_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil"); txt = str(ex.get("html", "") or ex.get("text", ""))
                    name = img_dir / f"pt_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "text": txt}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["pubtables_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["pubtables_error"] = str(e)
    return out


def fetch_objects365_images_list(limit: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = None
        try:
            ds = load_dataset("objects365", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            return out
        lst = Path("data/segmentation/objects365_images.txt"); _ensure_dir(lst.parent)
        n = 0
        with open(lst, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    if hasattr(img, "filename"):
                        f.write(str(Path(getattr(img, 'filename', '')).as_posix()) + "\n")
                        n += 1
                        if n >= max(1, int(limit)):
                            break
                except Exception:
                    continue
        if n > 0:
            out["objects365_images_list"] = str(lst.as_posix())
    except Exception as e:
        out["objects365_images_error"] = str(e)
    return out


def fetch_cc100_small(limit: int) -> dict[str, str]:
    """Fetch small multilingual text shards from CC-100 for a few languages.
    Writes text files under data/text/cc100_<lang>.txt to be picked up by DSM/KD.
    Controlled by env OMNICODER_MULTI_LANG_LANGS (comma list)."""
    out: dict[str, str] = {}
    try:
        langs_env = os.getenv("OMNICODER_MULTI_LANG_LANGS", "en,de,fr")
        langs = [x.strip() for x in langs_env.split(",") if x.strip()]
        from datasets import load_dataset  # type: ignore
        for lang in langs[:6]:
            try:
                ds = load_dataset("cc100", lang, split=f"train[:{max(1,limit)}]", streaming=True)
                outp = Path(f"data/text/cc100_{lang}.txt"); _ensure_dir(outp.parent)
                existing = _count_lines(outp)
                if existing < limit:
                    _log(f"append cc100[{lang}]: {existing} -> {limit}")
                    def _iter() -> Iterable[str]:
                        n = 0
                        for ex in ds:
                            txt = str(ex.get("text", ""))
                            if txt.strip():
                                yield txt
                                n += 1
                                if existing + n >= limit:
                                    break
                    _append_lines_until(outp, _iter(), limit)
                out[f"cc100_{lang}"] = str(outp)
            except Exception as e:
                out[f"cc100_{lang}_error"] = str(e)
    except Exception as e:
        out["cc100_error"] = str(e)
    return out


def fetch_mls_small(limit: int) -> dict[str, str]:
    """Fetch small MLS (Multilingual LibriSpeech) wavs+transcripts for one language.
    Writes data/asr/mls_wavs/ + data/asr/mls_transcripts.jsonl.
    Controlled by OMNICODER_MLS_LANG (default 'english')."""
    out: dict[str, str] = {}
    try:
        lang = os.getenv("OMNICODER_MLS_LANG", "english").strip()
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("mls", lang, split=f"train[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        wav_dir = Path("data/asr/mls_wavs"); _ensure_dir(wav_dir)
        tr_jsonl = Path("data/asr/mls_transcripts.jsonl")
        import json, soundfile as sf  # type: ignore
        with open(tr_jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                    text = str(ex.get("text", ""))
                    outp = wav_dir / f"mls_{i:05d}.wav"; sf.write(outp, arr, sr)
                    f.write(json.dumps({"path": str(outp.as_posix()), "text": text}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["mls_wavs_dir"] = str(wav_dir.as_posix())
        out["mls_transcripts"] = str(tr_jsonl.as_posix())
    except Exception as e:
        out["mls_error"] = str(e)
    return out


def fetch_ocr_synth(limit: int) -> dict[str, str]:
    """Synthesize a tiny OCR dataset: images with random text; write VQA-style JSONL."""
    out: dict[str, str] = {}
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        import random as _rand
        img_dir = Path("data/vqa/ocr_synth"); _ensure_dir(img_dir)
        jsonl = Path("data/vqa/ocr_synth.jsonl")
        import json
        words = ["alpha","bravo","charlie","delta","echo","foxtrot","zulu","lorem","ipsum","omnicoder"]
        with open(jsonl, "w", encoding="utf-8") as f:
            for i in range(max(1, min(200, limit))):
                w = 256; h = 256
                img = Image.new("RGB", (w, h), (255, 255, 255))
                dr = ImageDraw.Draw(img)
                txt = " ".join(_rand.sample(words, k=min(3, len(words))))
                x = _rand.randint(10, 60); y = _rand.randint(10, 200)
                dr.text((x, y), txt, fill=(0, 0, 0))
                p = img_dir / f"ocr_{i:04d}.png"
                img.save(p)
                f.write(json.dumps({"image": str(p.as_posix()), "question": "What text is shown?", "answer": txt}, ensure_ascii=False) + "\n")
        out["ocr_synth_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["ocr_synth_error"] = str(e)
    return out


def fetch_docvqa_small(limit: int) -> dict[str, str]:
    """Fetch a small DocVQA-style split; fall back to DocVQA subset if accessible on HF.
    Writes data/docvqa.jsonl with {image, question, answer}.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = None
        try:
            ds = load_dataset("docvqa/docvqa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["docvqa_error"] = "dataset unavailable"
            return out
        img_dir = Path("data/docvqa/images"); _ensure_dir(img_dir)
        jsonl = Path("data/docvqa/docvqa.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    q = str(ex.get("question", ""))
                    a = str(ex.get("answer", ""))
                    name = img_dir / f"doc_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["docvqa_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["docvqa_error"] = str(e)
    return out


def fetch_chartqa_small(limit: int) -> dict[str, str]:
    """Fetch a small ChartQA split and write VQA-style JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = None
        try:
            ds = load_dataset("chartqa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["chartqa_error"] = "dataset unavailable"
            return out
        img_dir = Path("data/chartqa/images"); _ensure_dir(img_dir)
        jsonl = Path("data/chartqa/chartqa.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    q = str(ex.get("question", ""))
                    a = str(ex.get("answer", ""))
                    name = img_dir / f"chart_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["chartqa_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["chartqa_error"] = str(e)
    return out


def fetch_flickr30k(limit: int) -> dict[str, str]:
    """Fetch Flickr30k subset; write images and captions JSONL compatible with VL trainers."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("flickr30k", split=f"train[:{max(1,limit)}]")
        img_dir = Path("data/flickr30k/images"); _ensure_dir(img_dir)
        jsonl = Path("data/flickr30k/captions.jsonl")
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image")
                    caps = ex.get("sentences") or ex.get("captions") or []
                    cap = ""
                    if isinstance(caps, list) and caps:
                        # sentences may be dicts {raw: str}
                        c0 = caps[0]
                        cap = str(c0.get("raw", "")) if isinstance(c0, dict) else str(c0)
                    name = img_dir / f"flk_{i:05d}.jpg"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["flickr30k_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["flickr30k_error"] = str(e)
    return out


def fetch_visual_genome(limit: int) -> dict[str, str]:
    """Fetch a tiny Visual Genome QA subset and materialize as VQA JSONL.
    Falls back to region descriptions if QA split is unavailable.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        qa = None
        try:
            # Some mirrors expose 'question_answers' config
            qa = load_dataset("visual_genome", "question_answers", split=f"train[:{max(1,limit)}]")
        except Exception:
            qa = None
        img_dir = Path("data/vqa/vg"); _ensure_dir(img_dir)
        jsonl = Path("data/vqa/vg_vqa.jsonl")
        import json
        if qa is not None:
            with open(jsonl, "w", encoding="utf-8") as f:
                for i, ex in enumerate(qa):
                    try:
                        img = ex.get("image") or ex.get("image_pil")
                        q = str(ex.get("question", ""))
                        a = str(ex.get("answer", ""))
                        name = img_dir / f"vg_{i:05d}.png"
                        if hasattr(img, "save"):
                            img.save(name)
                            f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                    except Exception:
                        continue
            out["vg_vqa_jsonl"] = str(jsonl.as_posix())
        else:
            # Fallback: build simple Q/A from region descriptions (treat description as answer)
            try:
                rd = load_dataset("visual_genome", "region_descriptions", split=f"train[:{max(1,limit)}]")
                with open(jsonl, "w", encoding="utf-8") as f:
                    for i, ex in enumerate(rd):
                        try:
                            img = ex.get("image") or ex.get("image_pil")
                            regions = ex.get("regions") or []
                            a = ""
                            if isinstance(regions, list) and regions:
                                r0 = regions[0]
                                a = str(r0.get("phrase", "")) if isinstance(r0, dict) else str(r0)
                            q = "What is described?"
                            name = img_dir / f"vg_{i:05d}.png"
                            if hasattr(img, "save"):
                                img.save(name)
                                f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                        except Exception:
                            continue
                out["vg_vqa_jsonl"] = str(jsonl.as_posix())
            except Exception:
                pass
    except Exception as e:
        out["visual_genome_error"] = str(e)
    return out


def fetch_coco_instances_small(limit: int) -> dict[str, str]:
    """If COCO 2017 instances annotations/images exist locally, materialize a small
    list of training images for grounding/segmentation-related passes.
    Writes an image list file and returns images_dir.
    """
    out: dict[str, str] = {}
    try:
        root = Path("data/coco2017")
        ann = root / "annotations" / "instances_train2017.json"
        imgs_dir = root / "train2017"
        if (not ann.exists()) or (not imgs_dir.exists()):
            return out
        import json
        data = json.loads(ann.read_text(encoding='utf-8'))
        images = data.get("images", [])
        lst = Path("data/segmentation/coco_images.txt")
        _ensure_dir(lst.parent)
        n = 0
        with open(lst, "w", encoding="utf-8") as f:
            for im in images:
                try:
                    name = im.get("file_name", "")
                    p = (imgs_dir / name)
                    if p.exists():
                        f.write(str(p.as_posix()) + "\n")
                        n += 1
                        if n >= max(1, int(limit)):
                            break
                except Exception:
                    continue
        if n > 0:
            out["coco_seg_images_dir"] = str(imgs_dir.as_posix())
            out["coco_seg_list"] = str(lst.as_posix())
    except Exception as e:
        out["coco_instances_small_error"] = str(e)
    return out


def fetch_lvis_small_images_list(limit: int) -> dict[str, str]:
    """If LVIS is accessible via HF or local mirrors, build an image list for seg/grounding."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = None
        try:
            ds = load_dataset("lvis", "v1", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            return out
        lst = Path("data/segmentation/lvis_images.txt"); _ensure_dir(lst.parent)
        n = 0
        with open(lst, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    name = None
                    if hasattr(img, "filename"):
                        name = getattr(img, "filename", None)
                    if name:
                        f.write(str(Path(name).as_posix()) + "\n")
                        n += 1
                        if n >= max(1, int(limit)):
                            break
                except Exception:
                    continue
        if n > 0:
            out["lvis_images_list"] = str(lst.as_posix())
    except Exception as e:
        out["lvis_small_error"] = str(e)
    return out


def fetch_openimages_small_images_list(limit: int) -> dict[str, str]:
    """Try OpenImages V6 small subset; write an images list for grounding/seg."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = None
        try:
            ds = load_dataset("open_images_v6", "validation", split=f"validation[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            return out
        lst = Path("data/segmentation/openimages_images.txt"); _ensure_dir(lst.parent)
        n = 0
        with open(lst, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    name = None
                    if hasattr(img, "filename"):
                        name = getattr(img, "filename", None)
                    if name:
                        f.write(str(Path(name).as_posix()) + "\n")
                        n += 1
                        if n >= max(1, int(limit)):
                            break
                except Exception:
                    continue
        if n > 0:
            out["openimages_images_list"] = str(lst.as_posix())
    except Exception as e:
        out["openimages_small_error"] = str(e)
    return out

def fetch_flickr8k_small(limit: int) -> dict[str, str]:
    """Fetch a tiny Flickr8k subset; write images and captions JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("flickr8k", split=f"train[:{max(1,limit)}]")
        img_dir = Path("data/flickr8k/images"); _ensure_dir(img_dir)
        jsonl = Path("data/flickr8k/captions.jsonl")
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image")
                    caps = ex.get("caption") or ex.get("captions") or []
                    cap = ""
                    if isinstance(caps, list) and caps:
                        cap = str(caps[0])
                    elif isinstance(caps, str):
                        cap = caps
                    name = img_dir / f"flk8_{i:05d}.jpg"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["flickr8k_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["flickr8k_error"] = str(e)
    return out


def fetch_vatex_small(limit: int) -> dict[str, str]:
    """Fetch a tiny VATEX subset; write mp4 clips and captions JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/vatex"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/vatex_captions.jsonl")
        ds = None
        try:
            ds = load_dataset("vatex", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["vatex_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    cap = str(ex.get("sentence") or ex.get("caption") or "")
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy()
                        outp = vid_dir / f"vatex_{i:05d}.mp4"
                        iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                        path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and cap:
                        f.write(json.dumps({"video": path, "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["vatex_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["vatex_error"] = str(e)
    return out


def fetch_webvid(limit: int) -> dict[str, str]:
    """Fetch WebVid metadata or small clips if available; writes a CSV of urls and a folder for clips.
    If actual video tensors are accessible via datasets, write MP4s under data/video/webvid/. Otherwise, emit CSV for later downloader.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        csv_path = Path("data/webvid/webvid_train.csv"); _ensure_dir(csv_path.parent)
        clips_dir = Path("data/video/webvid"); _ensure_dir(clips_dir)
        ds = None
        try:
            ds = load_dataset("webvid", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            # Try alternate namespace
            try:
                ds = load_dataset("webvid/webvid", split=f"train[:{max(1,limit)}]")
            except Exception:
                ds = None
        urls = []
        if ds is not None:
            # Attempt to persist URLs; if video array present, write small clips
            import csv
            import imageio.v2 as iio  # type: ignore
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_url", "caption"])  # header
                for i, ex in enumerate(ds):
                    try:
                        url = str(ex.get("video_url") or ex.get("url") or "")
                        cap = str(ex.get("caption", ""))
                        if url:
                            urls.append(url)
                            writer.writerow([url, cap])
                        v = ex.get("video")
                        if hasattr(v, "to_numpy"):
                            arr = v.to_numpy()
                            outp = clips_dir / f"webvid_{i:05d}.mp4"
                            iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                    except Exception:
                        continue
            out["webvid_csv"] = str(csv_path.as_posix())
            out["video_dir"] = str(clips_dir.as_posix())
    except Exception as e:
        out["webvid_error"] = str(e)
    return out


def fetch_audiocaps(limit: int) -> dict[str, str]:
    """Fetch AudioCaps subset; write WAVs and captions JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("audiocaps", split=f"train[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=32000))
        wav_dir = Path("data/audiocaps/wavs"); _ensure_dir(wav_dir)
        jsonl = Path("data/audiocaps/captions.jsonl")
        import json, soundfile as sf  # type: ignore
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                    cap = str(ex.get("caption", ""))
                    outp = wav_dir / f"ac_{i:05d}.wav"; sf.write(outp, arr, sr)
                    f.write(json.dumps({"path": str(outp.as_posix()), "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["ac_wavs_dir"] = str(wav_dir.as_posix())
        out["ac_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["audiocaps_error"] = str(e)
    return out


def fetch_msvdqa_small(limit: int) -> dict[str, str]:
    """Fetch a tiny MSVD-QA subset; write clips and QA JSONL when arrays accessible."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/msvdqa"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/msvd_qa.jsonl")
        ds = None
        try:
            ds = load_dataset("msvd_qa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["msvdqa_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    q = str(ex.get("question", "")); a = str(ex.get("answer", ""))
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"msvdqa_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and q:
                        f.write(json.dumps({"video": path, "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["msvdqa_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["msvdqa_error"] = str(e)
    return out


def fetch_tgifqa_small(limit: int) -> dict[str, str]:
    """Fetch a tiny TGIF-QA subset; write clips and QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/tgifqa"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/tgif_qa.jsonl")
        ds = None
        try:
            ds = load_dataset("tgif_qa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["tgifqa_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    q = str(ex.get("question", "")); a = str(ex.get("answer", ""))
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"tgifqa_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and q:
                        f.write(json.dumps({"video": path, "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["tgifqa_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["tgifqa_error"] = str(e)
    return out


def fetch_textvqa_small(limit: int) -> dict[str, str]:
    """Fetch a small TextVQA subset for OCR-VQA; write JSONL under data/textvqa.jsonl."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = None
        try:
            ds = load_dataset("textvqa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["textvqa_error"] = "dataset unavailable"
            return out
        img_dir = Path("data/textvqa/images"); _ensure_dir(img_dir)
        jsonl = Path("data/textvqa/textvqa.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    img = ex.get("image") or ex.get("image_pil")
                    q = str(ex.get("question", "")); a = str(ex.get("answer", ""))
                    name = img_dir / f"textvqa_{i:05d}.png"
                    if hasattr(img, "save"):
                        img.save(name)
                        f.write(json.dumps({"image": str(name.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["textvqa_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["textvqa_error"] = str(e)
    return out


def fetch_scienceqa_small(limit: int) -> dict[str, str]:
    """Fetch a small ScienceQA subset (vision subset) and materialize image+QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        img_dir = Path("data/scienceqa/images"); _ensure_dir(img_dir)
        jsonl = Path("data/scienceqa/scienceqa.jsonl")
        ds = None
        try:
            ds = load_dataset("scienceqa", "full", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["scienceqa_error"] = "dataset unavailable"
            return out
        with open(jsonl, "w", encoding="utf-8") as f:
            n = 0
            for ex in ds:
                try:
                    q = str(ex.get("question", ""))
                    a = str(ex.get("answer", ""))
                    img = ex.get("image") or ex.get("image_pil")
                    if hasattr(img, "save") and q and a:
                        p = img_dir / f"sqa_{n:05d}.png"; img.save(p)
                        f.write(json.dumps({"image": str(p.as_posix()), "question": q, "answer": a}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= limit:
                            break
                except Exception:
                    continue
        if n > 0:
            out["scienceqa_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["scienceqa_error"] = str(e)
    return out


def fetch_tvqa_small(limit: int) -> dict[str, str]:
    """Fetch a tiny TVQA subset; write mp4 clips when accessible and a QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/tvqa"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/tvqa_qa.jsonl")
        ds = None
        try:
            ds = load_dataset("tvqa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["tvqa_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    q = str(ex.get("question", "")); a = str(ex.get("answer", ""))
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"tvqa_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and q:
                        f.write(json.dumps({"video": path, "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["tvqa_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["tvqa_error"] = str(e)
    return out


def fetch_nextqa_small(limit: int) -> dict[str, str]:
    """Fetch a tiny NExT-QA subset; write clips and QA JSONL if arrays accessible."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/nextqa"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/nextqa_qa.jsonl")
        ds = None
        try:
            ds = load_dataset("nextqa", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["nextqa_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    q = str(ex.get("question", "")); a = str(ex.get("answer", ""))
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy(); outp = vid_dir / f"nextqa_{i:05d}.mp4"; iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25)); path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and q:
                        f.write(json.dumps({"video": path, "question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["nextqa_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["nextqa_error"] = str(e)
    return out


def fetch_msrvtt_small(limit: int) -> dict[str, str]:
    """Fetch a tiny MSR-VTT subset; write mp4 clips to data/video/msrvtt and a caption JSONL.
    Tries multiple dataset names and degrades gracefully if unavailable.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/msrvtt"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/msrvtt_captions.jsonl")
        ds = None
        try:
            ds = load_dataset("msr_vtt", split=f"train[:{max(1,limit)}]")
        except Exception:
            try:
                ds = load_dataset("msrvtt", split=f"train[:{max(1,limit)}]")
            except Exception:
                ds = None
        if ds is None:
            out["msrvtt_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    cap = str(ex.get("sentence") or ex.get("caption") or "")
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy()
                        outp = vid_dir / f"msrvtt_{i:05d}.mp4"
                        iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                        path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and cap:
                        f.write(json.dumps({"video": path, "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["msrvtt_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["msrvtt_error"] = str(e)
    return out


def fetch_youcook2_small(limit: int) -> dict[str, str]:
    """Fetch a tiny YouCook2 subset; write clips to data/video/youcook2 and captions JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/youcook2"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/youcook2_captions.jsonl")
        ds = None
        try:
            ds = load_dataset("youcook2", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["youcook2_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    cap = str(ex.get("caption") or ex.get("text") or "")
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy()
                        outp = vid_dir / f"yc2_{i:05d}.mp4"
                        iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                        path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and cap:
                        f.write(json.dumps({"video": path, "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["youcook2_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["youcook2_error"] = str(e)
    return out


def fetch_didemo_small(limit: int) -> dict[str, str]:
    """Fetch a tiny DiDeMo subset; write clips and captions JSONL if arrays are accessible."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/didemo"); _ensure_dir(vid_dir)
        jsonl = Path("data/video/didemo_captions.jsonl")
        ds = None
        try:
            ds = load_dataset("didemo", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["didemo_error"] = "dataset unavailable"
            return out
        import json
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    caps = ex.get("captions") or []
                    cap = str(caps[0]) if isinstance(caps, list) and caps else ""
                    v = ex.get("video")
                    path = None
                    if hasattr(v, "to_numpy"):
                        arr = v.to_numpy()
                        outp = vid_dir / f"didemo_{i:05d}.mp4"
                        iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
                        path = str(outp.as_posix())
                    elif isinstance(v, str) and v:
                        path = v
                    if path and cap:
                        f.write(json.dumps({"video": path, "text": cap}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["didemo_jsonl"] = str(jsonl.as_posix())
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["didemo_error"] = str(e)
    return out
def fetch_hmdb51_small(limit: int) -> dict[str, str]:
    """Fetch a tiny HMDB51 subset; write clips to data/video/hmdb51."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/hmdb51"); _ensure_dir(vid_dir)
        ds = load_dataset("hmdb51", split=f"train[:{max(1,limit)}]")
        for i, ex in enumerate(ds):
            try:
                v = ex.get("video")
                if hasattr(v, "to_numpy"):
                    arr = v.to_numpy()
                    outp = vid_dir / f"hmdb_{i:05d}.mp4"
                    iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
            except Exception:
                continue
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["hmdb51_error"] = str(e)
    return out


def fetch_cifar10_images(limit: int) -> dict[str, str]:
    """Fetch CIFAR-10 images; write to data/vl/images_cifar for extra visual variety."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import numpy as _np  # type: ignore
        from PIL import Image  # type: ignore
        ds = load_dataset("cifar10", split=f"train[:{max(1,limit)}]")
        img_dir = Path("data/vl/images_cifar"); _ensure_dir(img_dir)
        for i, ex in enumerate(ds):
            try:
                arr = _np.array(ex.get("img"))
                p = img_dir / f"cifar_{i:05d}.png"
                Image.fromarray(arr).save(p)
            except Exception:
                continue
        out["cifar_images_dir"] = str(img_dir.as_posix())
    except Exception as e:
        out["cifar10_error"] = str(e)
    return out


def fetch_text_qa_small(limit: int) -> dict[str, str]:
    """Create small JSONLs for ARC, TruthfulQA, Winogrande, HellaSwag, MMLU (subset) for QA finetune.
    Writes under data/qa/*.jsonl and returns env keys mapping.
    """
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        # ARC Easy
        try:
            ds = load_dataset("ai2_arc", "ARC-Easy", split=f"train[:{max(1,limit)}]")
            p = Path("data/qa/arc_easy.jsonl"); _ensure_dir(p.parent)
            with open(p, "w", encoding="utf-8") as f:
                for ex in ds:
                    q = ex.get("question", ""); ans = ex.get("answerKey", "")
                    f.write(json.dumps({"question": q, "answer": ans}, ensure_ascii=False) + "\n")
            out["arc_jsonl"] = str(p)
        except Exception as e:
            out["arc_error"] = str(e)
        # TruthfulQA
        try:
            ds = load_dataset("truthful_qa", "generation", split=f"validation[:{max(1,limit)}]")
            p = Path("data/qa/truthfulqa.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for ex in ds:
                    q = ex.get("question", "");
                    ans = ", ".join(ex.get("best_answer", [])) if isinstance(ex.get("best_answer"), list) else str(ex.get("best_answer", ""))
                    f.write(json.dumps({"question": q, "answer": ans}, ensure_ascii=False) + "\n")
            out["truthfulqa_jsonl"] = str(p)
        except Exception as e:
            out["truthfulqa_error"] = str(e)
        # Winogrande
        try:
            ds = load_dataset("winogrande", "winogrande_xl", split=f"train[:{max(1,limit)}]")
            p = Path("data/qa/winogrande.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for ex in ds:
                    q = ex.get("sentence", "").replace("_", "____")
                    a = str(ex.get("answer", ""))
                    f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            out["winogrande_jsonl"] = str(p)
        except Exception as e:
            out["winogrande_error"] = str(e)
        # HellaSwag
        try:
            ds = load_dataset("hellaswag", split=f"train[:{max(1,limit)}]")
            p = Path("data/qa/hellaswag.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for ex in ds:
                    q = ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")
                    labels = ex.get("endings", [])
                    a = str(labels[int(ex.get("label", 0))]) if labels else ""
                    f.write(json.dumps({"question": q.strip(), "answer": a}, ensure_ascii=False) + "\n")
            out["hellaswag_jsonl"] = str(p)
        except Exception as e:
            out["hellaswag_error"] = str(e)
        # MMLU (subset fallback: pick one subject) — optional; can be large
        try:
            subj = os.getenv("OMNICODER_MMLU_SUBJECT", "abstract_algebra")
            ds = load_dataset("cais/mmlu", subj, split=f"test[:{max(1,limit)}]")
            p = Path("data/qa/mmlu.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for ex in ds:
                    q = ex.get("question", ""); a = ex.get("answer", "")
                    f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            out["mmlu_jsonl"] = str(p)
        except Exception as e:
            out["mmlu_error"] = str(e)
    except Exception as e:
        out["text_qa_error"] = str(e)
    return out


def fetch_agieval_small(limit: int) -> dict[str, str]:
    """Fetch a tiny AGIEval subset and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        # Try several configs; pick the first available
        cfgs = ["gaokao_chinese", "logiqa2", "aqua_rat"]
        for cfg in cfgs:
            try:
                ds = load_dataset("agieval", cfg, split=f"validation[:{max(1,limit)}]")
                p = Path("data/qa/agieval.jsonl"); _ensure_dir(p.parent)
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        q = ex.get("question", "")
                        a = ex.get("answer", "")
                        f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                out["agieval_jsonl"] = str(p)
                return out
            except Exception:
                continue
        out["agieval_error"] = "no config accessible"
    except Exception as e:
        out["agieval_error"] = str(e)
    return out


def fetch_bbh_small(limit: int) -> dict[str, str]:
    """Fetch a tiny BBH subset (date_understanding) and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        try:
            ds = load_dataset("google/bbh", "date_understanding", split=f"test[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["bbh_error"] = "dataset unavailable"
            return out
        p = Path("data/qa/bbh.jsonl"); _ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = ex.get("input", "")
                    a = ex.get("target", "")
                    f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["bbh_jsonl"] = str(p)
    except Exception as e:
        out["bbh_error"] = str(e)
    return out


def fetch_strategyqa_small(limit: int) -> dict[str, str]:
    """Fetch StrategyQA small subset and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = load_dataset("strategyqa", split=f"train[:{max(1,limit)}]")
        p = Path("data/qa/strategyqa.jsonl"); _ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = ex.get("question", "")
                    a = ex.get("answer", "")
                    f.write(json.dumps({"question": q, "answer": str(a)}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["strategyqa_jsonl"] = str(p)
    except Exception as e:
        out["strategyqa_error"] = str(e)
    return out


def fetch_arc_challenge(limit: int) -> dict[str, str]:
    """Fetch ARC-Challenge subset and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=f"train[:{max(1,limit)}]")
        p = Path("data/qa/arc_ch.jsonl"); _ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = ex.get("question", ""); ans = ex.get("answerKey", "")
                    f.write(json.dumps({"question": q, "answer": ans}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["arc_ch_jsonl"] = str(p)
    except Exception as e:
        out["arc_ch_error"] = str(e)
    return out


def fetch_nq_small(limit: int) -> dict[str, str]:
    """Fetch Natural Questions (validation subset) small and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = load_dataset("natural_questions", split=f"validation[:{max(1,limit)}]")
        p = Path("data/qa/nq.jsonl"); _ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = ex.get("question", "")
                    # Prefer short answers if available
                    ans = ex.get("answers", {})
                    a = ""
                    if isinstance(ans, dict):
                        sa = ans.get("text", [])
                        a = str(sa[0]) if sa else ""
                    f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["nq_jsonl"] = str(p)
    except Exception as e:
        out["nq_error"] = str(e)
    return out


def fetch_hotpotqa_small(limit: int) -> dict[str, str]:
    """Fetch HotpotQA small subset and write QA JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import json
        ds = load_dataset("hotpot_qa", "full_wiki", split=f"validation[:{max(1,limit)}]")
        p = Path("data/qa/hotpotqa.jsonl"); _ensure_dir(p.parent)
        with open(p, "w", encoding="utf-8") as f:
            for ex in ds:
                try:
                    q = ex.get("question", "")
                    a = ex.get("answer", "")
                    f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["hotpotqa_jsonl"] = str(p)
    except Exception as e:
        out["hotpotqa_error"] = str(e)
    return out
def fetch_esc50_small(limit: int) -> dict[str, str]:
    """Fetch a tiny ESC-50 audio events subset; write WAVs and labels JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("esc50", split=f"train[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        wav_dir = Path("data/audio/esc50"); _ensure_dir(wav_dir)
        jsonl = Path("data/audio/esc50.jsonl")
        import json, soundfile as sf  # type: ignore
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                    cat = str(ex.get("category", ""))
                    outp = wav_dir / f"esc_{i:05d}.wav"; sf.write(outp, arr, sr)
                    f.write(json.dumps({"path": str(outp.as_posix()), "label": cat}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["esc50_wavs_dir"] = str(wav_dir.as_posix())
        out["esc50_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["esc50_error"] = str(e)
    return out


def fetch_vggsound_small(limit: int) -> dict[str, str]:
    """Fetch a tiny VGGSound subset if available; write WAVs and labels JSONL."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset, Audio  # type: ignore
        ds = load_dataset("vggsound", split=f"train[:{max(1,limit)}]")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        wav_dir = Path("data/audio/vggsound"); _ensure_dir(wav_dir)
        jsonl = Path("data/audio/vggsound.jsonl")
        import json, soundfile as sf  # type: ignore
        with open(jsonl, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                try:
                    arr = ex["audio"]["array"]; sr = int(ex["audio"]["sampling_rate"])
                    cat = str(ex.get("category", "") or ex.get("label", ""))
                    outp = wav_dir / f"vgg_{i:05d}.wav"; sf.write(outp, arr, sr)
                    f.write(json.dumps({"path": str(outp.as_posix()), "label": cat}, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        out["vggsound_wavs_dir"] = str(wav_dir.as_posix())
        out["vggsound_jsonl"] = str(jsonl.as_posix())
    except Exception as e:
        out["vggsound_error"] = str(e)
    return out


def fetch_ssv2_small(limit: int) -> dict[str, str]:
    """Fetch Something-Something V2 small subset; write mp4 clips to data/video/ssv2."""
    out: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        import imageio.v2 as iio  # type: ignore
        vid_dir = Path("data/video/ssv2"); _ensure_dir(vid_dir)
        ds = None
        try:
            ds = load_dataset("something_something_v2", split=f"train[:{max(1,limit)}]")
        except Exception:
            ds = None
        if ds is None:
            out["ssv2_error"] = "dataset unavailable"
            return out
        for i, ex in enumerate(ds):
            try:
                v = ex.get("video")
                if hasattr(v, "to_numpy"):
                    arr = v.to_numpy()
                    outp = vid_dir / f"ssv2_{i:05d}.mp4"
                    iio.mimwrite(outp, arr, fps=getattr(v, "fps", 25))
            except Exception:
                continue
        out["video_dir"] = str(vid_dir.as_posix())
    except Exception as e:
        out["ssv2_small_error"] = str(e)
    return out


def fetch_sr_seed(limit: int) -> dict[str, str]:
    """Create a tiny Super-Resolution seed dataset with matching pred/ref dirs.
    Generates synthetic images and their downsampled counterparts so SR training can run OOTB.
    """
    out: dict[str, str] = {}
    try:
        import numpy as _np  # type: ignore
        from pathlib import Path as _P
        ref_dir = _P("data/sr/ref"); pred_dir = _P("data/sr/pred")
        _ensure_dir(ref_dir); _ensure_dir(pred_dir)
        def _write_img(arr: "Any", path: _P) -> None:
            try:
                from PIL import Image  # type: ignore
                Image.fromarray(arr).save(path)
                return
            except Exception:
                pass
            try:
                import imageio.v2 as iio  # type: ignore
                iio.imwrite(path, arr)
            except Exception:
                pass
        n = max(1, min(4, int(limit)))
        for i in range(n):
            H, W = 256, 256
            base = _np.linspace(0, 255, W, dtype=_np.uint8)[None, :].repeat(H, axis=0)
            img = _np.stack([
                base,
                _np.roll(base, i * 4, axis=1),
                _np.roll(base, i * 8, axis=0)
            ], axis=-1)
            ref_path = ref_dir / f"sr_ref_{i:03d}.png"
            _write_img(img, ref_path)
            try:
                from PIL import Image  # type: ignore
                lr = Image.fromarray(img).resize((W // 4, H // 4), resample=Image.BICUBIC)
                lr = lr.resize((W, H), resample=Image.BICUBIC)
                pred_img = _np.array(lr)
            except Exception:
                pred_img = img[::4, ::4]
                pred_img = pred_img.repeat(4, axis=0).repeat(4, axis=1)[:H, :W]
            pred_path = pred_dir / f"sr_pred_{i:03d}.png"
            _write_img(pred_img, pred_path)
        out["sr_ref_dir"] = str(ref_dir.as_posix())
        out["sr_pred_dir"] = str(pred_dir.as_posix())
    except Exception as e:
        out["sr_seed_error"] = str(e)
    return out


def autofetch_all(limit: int) -> dict[str, Any]:
    """Programmatic API: fetch a comprehensive suite of datasets across modalities.
    Returns a dict of produced file/dir paths and any *_error keys.
    """
    limit = max(1, int(limit))
    out: dict[str, Any] = {}

    try:
        out["text"] = fetch_text(limit)
    except Exception as e:
        out["text_error"] = str(e)
    try:
        out.update(fetch_text_more(limit))
    except Exception as e:
        out["text_more_error"] = str(e)
    try:
        out["code"] = fetch_code(min(limit, 1000))
    except Exception as e:
        out["code_error"] = str(e)
    try:
        out.update(fetch_code_more(min(limit, 2000)))
    except Exception as e:
        out["code_more_error"] = str(e)
    try:
        # Skip VL fetch if files already materialized to avoid repeating costly downloads
        vl_done = Path("data/vl_cc.jsonl").exists() or Path("data/vl_coco.jsonl").exists() or Path("data/coco2017/train2017").exists()
        if not vl_done or os.getenv('OMNICODER_FORCE_FETCH', '0') == '1':
            out.update(fetch_vl(min(limit, 1000)))
    except Exception as e:
        out["vl_error"] = str(e)
    try:
        # Skip regenerate when jsonl already present; only extend when force is set
        vl_more_done = Path("data/vl_cc.jsonl").exists() or Path("data/vl_coco.jsonl").exists()
        if (not vl_more_done) or os.getenv('OMNICODER_FORCE_FETCH','0')=='1':
            out.update(fetch_vl_more(min(limit, 3000)))
    except Exception as e:
        out["vl_more_error"] = str(e)
    try:
        vqa_done = Path("data/vqa").exists() or Path("examples/vqa_fused_sample.jsonl").exists()
        if (not vqa_done) or os.getenv('OMNICODER_FORCE_FETCH','0')=='1':
            out.update(fetch_vqa(min(limit, 2000)))
    except Exception as e:
        out["vqa_error"] = str(e)
    try:
        # Persist ASR outputs; skip refetch if outputs exist
        asr_done = Path("data/asr/wavs").exists() and Path("data/asr/transcripts.jsonl").exists()
        if not asr_done or os.getenv('OMNICODER_FORCE_FETCH', '0') == '1':
            out.update(fetch_asr(min(limit, 200)))
    except Exception as e:
        out["asr_error"] = str(e)
    try:
        tts_done = Path("data/tts/texts.txt").exists()
        if (not tts_done) or os.getenv('OMNICODER_FORCE_FETCH','0')=='1':
            out.update(fetch_tts(min(limit, 200)))
    except Exception as e:
        out["tts_error"] = str(e)
    try:
        video_done = Path("data/video").exists()
        if (not video_done) or os.getenv('OMNICODER_FORCE_FETCH','0')=='1':
            out.update(fetch_video(min(limit, 100)))
    except Exception as e:
        out["video_error"] = str(e)
    # Additional audio/VQA datasets
    try:
        out.update(fetch_speech_commands(min(limit, 200)))
    except Exception as e:
        out["speech_commands_error"] = str(e)
    try:
        out.update(fetch_textcaps(min(limit, 2000)))
    except Exception as e:
        out["textcaps_error"] = str(e)
    # Flickr30k captions
    try:
        out.update(fetch_flickr30k(min(limit, 3000)))
    except Exception as e:
        out["flickr30k_error"] = str(e)
    # Visual Genome VQA-like
    try:
        out.update(fetch_visual_genome(min(limit, 2000)))
    except Exception as e:
        out["visual_genome_error"] = str(e)
    # WebVid metadata/clips
    try:
        out.update(fetch_webvid(min(limit, 200)))
    except Exception as e:
        out["webvid_error"] = str(e)
    # MSR-VTT video+captions small
    try:
        out.update(fetch_msrvtt_small(min(limit, 100)))
    except Exception as e:
        out["msrvtt_small_error"] = str(e)
    # MSVD-QA and TGIF-QA small
    try:
        out.update(fetch_msvdqa_small(min(limit, 100)))
    except Exception as e:
        out["msvdqa_small_error"] = str(e)
    try:
        out.update(fetch_tgifqa_small(min(limit, 100)))
    except Exception as e:
        out["tgifqa_small_error"] = str(e)
    # YouCook2 video+captions small
    try:
        out.update(fetch_youcook2_small(min(limit, 100)))
    except Exception as e:
        out["youcook2_small_error"] = str(e)
    # DiDeMo video+captions small
    try:
        out.update(fetch_didemo_small(min(limit, 100)))
    except Exception as e:
        out["didemo_small_error"] = str(e)
    # HMDB51 action clips small
    try:
        out.update(fetch_hmdb51_small(min(limit, 100)))
    except Exception as e:
        out["hmdb51_small_error"] = str(e)
    # TVQA and NExT-QA small
    try:
        out.update(fetch_tvqa_small(min(limit, 100)))
    except Exception as e:
        out["tvqa_small_error"] = str(e)
    try:
        out.update(fetch_nextqa_small(min(limit, 100)))
    except Exception as e:
        out["nextqa_small_error"] = str(e)
    # CIFAR-10 image variety
    try:
        out.update(fetch_cifar10_images(min(limit, 1000)))
    except Exception as e:
        out["cifar10_small_error"] = str(e)
    # Flickr8k captions
    try:
        out.update(fetch_flickr8k_small(min(limit, 3000)))
    except Exception as e:
        out["flickr8k_small_error"] = str(e)
    # VATEX video+captions small
    try:
        out.update(fetch_vatex_small(min(limit, 100)))
    except Exception as e:
        out["vatex_small_error"] = str(e)
    # COCO instances grounding list if local artifacts exist
    try:
        out.update(fetch_coco_instances_small(min(limit, 500)))
    except Exception as e:
        out["coco_instances_small_error"] = str(e)
    # LVIS/OpenImages images lists
    try:
        out.update(fetch_lvis_small_images_list(min(limit, 500)))
    except Exception as e:
        out["lvis_images_small_error"] = str(e)
    try:
        out.update(fetch_openimages_small_images_list(min(limit, 500)))
    except Exception as e:
        out["openimages_images_small_error"] = str(e)
    # RefCOCO grounding JSONL (small)
    try:
        out.update(fetch_refcoco_small(min(limit, 2000)))
    except Exception as e:
        out["refcoco_small_error"] = str(e)
    # Text QA small
    try:
        out.update(fetch_text_qa_small(min(limit, 5000)))
    except Exception as e:
        out["text_qa_small_error"] = str(e)
    # Additional reasoning QA datasets
    try:
        out.update(fetch_agieval_small(min(limit, 2000)))
    except Exception as e:
        out["agieval_small_error"] = str(e)
    try:
        out.update(fetch_bbh_small(min(limit, 2000)))
    except Exception as e:
        out["bbh_small_error"] = str(e)
    try:
        out.update(fetch_strategyqa_small(min(limit, 5000)))
    except Exception as e:
        out["strategyqa_small_error"] = str(e)
    try:
        out.update(fetch_arc_challenge(min(limit, 5000)))
    except Exception as e:
        out["arc_ch_small_error"] = str(e)
    # Open-domain QA datasets
    try:
        out.update(fetch_nq_small(min(limit, 3000)))
    except Exception as e:
        out["nq_small_error"] = str(e)
    try:
        out.update(fetch_hotpotqa_small(min(limit, 3000)))
    except Exception as e:
        out["hotpotqa_small_error"] = str(e)
    # Audio events (ESC-50)
    try:
        out.update(fetch_esc50_small(min(limit, 200)))
    except Exception as e:
        out["esc50_small_error"] = str(e)
    # VGGSound small
    try:
        out.update(fetch_vggsound_small(min(limit, 200)))
    except Exception as e:
        out["vggsound_small_error"] = str(e)
    # Clotho audio captions small
    try:
        out.update(fetch_clotho_small(min(limit, 1000)))
    except Exception as e:
        out["clotho_small_error"] = str(e)
    # FSD50K events small
    try:
        out.update(fetch_fsd50k_small(min(limit, 1000)))
    except Exception as e:
        out["fsd50k_small_error"] = str(e)
    # Something-Something V2 small
    try:
        out.update(fetch_ssv2_small(min(limit, 100)))
    except Exception as e:
        out["ssv2_small_error"] = str(e)
    # AudioCaps
    try:
        out.update(fetch_audiocaps(min(limit, 1000)))
    except Exception as e:
        out["audiocaps_error"] = str(e)
    # Multilingual text (CC-100 small)
    try:
        out.update(fetch_cc100_small(min(limit, 20000)))
    except Exception as e:
        out["cc100_small_error"] = str(e)
    # Multilingual ASR (MLS small)
    try:
        out.update(fetch_mls_small(min(limit, 200)))
    except Exception as e:
        out["mls_small_error"] = str(e)
    # OCR synthetic (small)
    try:
        out.update(fetch_ocr_synth(min(limit, 500)))
    except Exception as e:
        out["ocr_synth_small_error"] = str(e)
    # DocVQA + ChartQA
    try:
        out.update(fetch_docvqa_small(min(limit, 1000)))
    except Exception as e:
        out["docvqa_small_error"] = str(e)
    try:
        out.update(fetch_chartqa_small(min(limit, 1000)))
    except Exception as e:
        out["chartqa_small_error"] = str(e)
    # TextVQA
    try:
        out.update(fetch_textvqa_small(min(limit, 1000)))
    except Exception as e:
        out["textvqa_small_error"] = str(e)
    # ScienceQA
    try:
        out.update(fetch_scienceqa_small(min(limit, 1000)))
    except Exception as e:
        out["scienceqa_small_error"] = str(e)
    # SR seed pairs
    try:
        out.update(fetch_sr_seed(min(limit, 8)))
    except Exception as e:
        out["sr_seed_error"] = str(e)
    # Synthetic multimodal instruction JSONL
    try:
        out.update(build_mm_instructions_small(min(limit, 2000)))
    except Exception as e:
        out["mm_instructions_small_error"] = str(e)
    # Real instruction datasets (small slices)
    try:
        out.update(fetch_llava_instruct_small(min(limit, 1000)))
    except Exception as e:
        out["llava_instruct_small_error"] = str(e)
    try:
        out.update(fetch_sharegpt4v_small(min(limit, 1000)))
    except Exception as e:
        out["sharegpt4v_small_error"] = str(e)
    try:
        out.update(fetch_minigpt4_instruct_small(min(limit, 1000)))
    except Exception as e:
        out["minigpt4_small_error"] = str(e)
    try:
        out.update(fetch_videochat_instruct_small(min(limit, 300)))
    except Exception as e:
        out["videochat_small_error"] = str(e)
    # Video temporal localization sources
    try:
        out.update(fetch_activitynet_captions_small(min(limit, 200)))
    except Exception as e:
        out["activitynet_captions_small_error"] = str(e)
    try:
        out.update(fetch_charades_sta_small(min(limit, 200)))
    except Exception as e:
        out["charades_sta_small_error"] = str(e)
    try:
        out.update(fetch_ego4d_nlq_small(min(limit, 200)))
    except Exception as e:
        out["ego4d_nlq_small_error"] = str(e)
    # PubTables and Objects365
    try:
        out.update(fetch_pubtables1m_small(min(limit, 500)))
    except Exception as e:
        out["pubtables1m_small_error"] = str(e)
    try:
        out.update(fetch_objects365_images_list(min(limit, 500)))
    except Exception as e:
        out["objects365_images_small_error"] = str(e)

    # Additional modalities: reasoning, math, tool/function-calling, music, python-specific code
    try:
        def fetch_reasoning(limit: int) -> dict[str, str]:
            out_r: dict[str, str] = {}
            from datasets import load_dataset  # type: ignore
            import json
            # GSM8K
            try:
                ds = load_dataset("gsm8k", "main", split=f"train[:{max(1,limit)}]")
                p = Path("data/reasoning/gsm8k.jsonl"); _ensure_dir(p.parent)
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({"question": ex.get("question", ""), "answer": ex.get("answer", "")}, ensure_ascii=False) + "\n")
                out_r["gsm8k"] = str(p)
            except Exception as e:  # pragma: no cover
                out_r["gsm8k_error"] = str(e)
            # SVAMP
            try:
                ds = load_dataset("svamp", split=f"train[:{max(1,limit)}]")
                p = Path("data/reasoning/svamp.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        q = ex.get("Body", "") + " " + ex.get("Question", "")
                        a = str(ex.get("Answer", ""))
                        f.write(json.dumps({"question": q.strip(), "answer": a}, ensure_ascii=False) + "\n")
                out_r["svamp"] = str(p)
            except Exception as e:
                out_r["svamp_error"] = str(e)
            # CommonsenseQA
            try:
                ds = load_dataset("commonsense_qa", split=f"train[:{max(1,limit)}]")
                p = Path("data/reasoning/commonsenseqa.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({"question": ex.get("question", ""), "choices": ex.get("choices", {}), "answerKey": ex.get("answerKey", "")}, ensure_ascii=False) + "\n")
                out_r["commonsenseqa"] = str(p)
            except Exception as e:
                out_r["commonsenseqa_error"] = str(e)
            return out_r

        out.update(fetch_reasoning(min(limit, 20000)))
    except Exception as e:
        out["reasoning_error"] = str(e)

    try:
        def fetch_math(limit: int) -> dict[str, str]:
            out_m: dict[str, str] = {}
            from datasets import load_dataset  # type: ignore
            import json
            # Hendrycks MATH
            try:
                ds = load_dataset("hendrycks/Math", split=f"train[:{max(1,limit)}]")
                p = Path("data/math/hendrycks_math.jsonl"); _ensure_dir(p.parent)
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({"problem": ex.get("problem", ""), "solution": ex.get("solution", "")}, ensure_ascii=False) + "\n")
                out_m["hendrycks_math"] = str(p)
            except Exception as e:
                out_m["hendrycks_math_error"] = str(e)
            # MATHQA
            try:
                ds = load_dataset("math_qa", split=f"train[:{max(1,limit)}]")
                p = Path("data/math/mathqa.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({"question": ex.get("Problem", ""), "options": ex.get("options", ""), "correct": ex.get("correct", "")}, ensure_ascii=False) + "\n")
                out_m["mathqa"] = str(p)
            except Exception as e:
                out_m["mathqa_error"] = str(e)
            # GSM8K test-only copy for eval
            try:
                ds = load_dataset("gsm8k", "main", split=f"test[:{max(1,limit)}]")
                p = Path("data/math/gsm8k_test.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps({"question": ex.get("question", ""), "answer": ex.get("answer", "")}, ensure_ascii=False) + "\n")
                out_m["gsm8k_test"] = str(p)
            except Exception as e:
                out_m["gsm8k_test_error"] = str(e)
            return out_m

        out.update(fetch_math(min(limit, 20000)))
    except Exception as e:
        out["math_error"] = str(e)

    try:
        def fetch_function_calls(limit: int) -> dict[str, str]:
            out_fc: dict[str, str] = {}
            from datasets import load_dataset  # type: ignore
            import json
            # Gorilla openfunctions (API/Tool calling style)
            try:
                ds = load_dataset("gorilla-llm/gorilla-openfunctions-v1", split=f"train[:{max(1,limit)}]")
                p = Path("data/tools/gorilla_openfunctions.jsonl"); _ensure_dir(p.parent)
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                out_fc["gorilla_openfunctions_v1"] = str(p)
            except Exception as e:
                out_fc["gorilla_openfunctions_error"] = str(e)
            # ToolBench (if accessible without token)
            try:
                ds = load_dataset("THUDM/ToolBench", split=f"train[:{max(1,limit)}]")
                p = Path("data/tools/toolbench.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                out_fc["toolbench"] = str(p)
            except Exception as e:
                out_fc["toolbench_error"] = str(e)
            # Function-calling synthetic prompts (if available)
            try:
                ds = load_dataset("m-a-p/FunctionCall", split=f"train[:{max(1,limit)}]")
                p = Path("data/tools/function_call.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for ex in ds:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                out_fc["function_call"] = str(p)
            except Exception as e:
                out_fc["function_call_error"] = str(e)
            return out_fc

        out.update(fetch_function_calls(min(limit, 50000)))
    except Exception as e:
        out["function_calls_error"] = str(e)

    try:
        def fetch_music(limit: int) -> dict[str, str]:
            out_mu: dict[str, str] = {}
            # Maestro v3 (official URL via TFDS mirror)
            try:
                url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
                dest = Path("data/music/maestro-v3.0.0.zip")
                if _aria2c(url, dest):
                    out_mu["maestro_zip"] = str(dest)
            except Exception as e:
                out_mu["maestro_error"] = str(e)
            # MusicNet (UW mirror) is frequently unavailable; use MusiCNN mirrors if provided, otherwise skip silently
            try:
                url = os.environ.get("OMNICODER_MUSICNET_URL", "")
                if url:
                    dest = Path("data/music/musicnet.zip")
                    if _aria2c(url, dest):
                        out_mu["musicnet_zip"] = str(dest)
            except Exception as e:
                out_mu["musicnet_error"] = str(e)
            # Lakh MIDI (LMD full)
            try:
                url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
                dest = Path("data/music/lmd_full.tar.gz")
                if _aria2c(url, dest):
                    out_mu["lakh_midi_tar"] = str(dest)
            except Exception as e:
                out_mu["lakh_midi_error"] = str(e)
            return out_mu

        out.update(fetch_music(min(limit, 1000000)))
    except Exception as e:
        out["music_error"] = str(e)

    try:
        def fetch_python_only(limit: int) -> dict[str, str]:
            out_py: dict[str, str] = {}
            from datasets import load_dataset  # type: ignore
            import json
            try:
                ds = load_dataset("bigcode/the-stack-smol", data_files="data/python/**/*.py", split="train", streaming=True)
                p = Path("data/code/python_stack.jsonl"); _ensure_dir(p.parent)
                with open(p, "w", encoding="utf-8") as f:
                    n = 0
                    for ex in ds:
                        content = ex.get("content") or ex.get("text") or ""
                        if content:
                            f.write(json.dumps({"lang": "python", "code": content}, ensure_ascii=False) + "\n")
                            n += 1
                            if n >= limit:
                                break
                out_py["python_stack_smol"] = str(p)
            except Exception as e:
                out_py["python_stack_error"] = str(e)
            return out_py

        out.update(fetch_python_only(min(limit, 200000)))
    except Exception as e:
        out["python_only_error"] = str(e)

    # Optionally fetch external mirrors (tokenless, user-provided URLs)
    try:
        out.update(fetch_external_archives())
    except Exception:
        pass

    # Official tokenless pulls from primary hosts (COCO, LibriSpeech, LJSpeech, WIT shard)
    try:
        out.update(fetch_coco_official(min(limit, 100000)))
    except Exception as e:
        out["coco_official_error"] = str(e)
    try:
        out.update(fetch_librispeech_official(min(limit, 10000)))
    except Exception as e:
        out["librispeech_official_error"] = str(e)
    try:
        out.update(fetch_ljspeech_official(min(limit, 10000)))
    except Exception as e:
        out["ljspeech_official_error"] = str(e)
    try:
        out.update(fetch_wit_official(min(limit, 200000)))
    except Exception as e:
        out["wit_official_error"] = str(e)
    # Additional manifests via public mirrors when provided
    try:
        out.update(fetch_cc12m_manifest())
    except Exception:
        pass
    try:
        out.update(fetch_coyo_manifest())
    except Exception:
        pass
    try:
        out.update(fetch_laion_shards())
    except Exception:
        pass

    # Best-effort: create lightweight eval JSONLs for VQAv2/OK-VQA/COCO/MSRVTT if datasets available
    try:
        def _write_pairs(pairs: list[dict], path: str) -> None:
            import json
            from pathlib import Path
            p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                for ex in pairs:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        # VQAv2
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("visual_qa_v2", split=f"validation[:{min(limit, 500)}]")
            pairs = []
            for ex in ds:
                img = ex.get('image')
                q = ex.get('question')
                a = ex.get('answers', {}).get('text', [""])
                if hasattr(img, 'filename'):
                    pairs.append({"image": getattr(img, 'filename', ''), "question": q, "answer": a[0] if a else ""})
            if pairs:
                _write_pairs(pairs, "data/vqa/vqav2_eval.jsonl")
                out["vqav2_jsonl"] = "data/vqa/vqav2_eval.jsonl"
        except Exception:
            pass
        # OK-VQA (if accessible)
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("ok_vqa", split=f"validation[:{min(limit, 500)}]")
            pairs = []
            for ex in ds:
                img = ex.get('image')
                q = ex.get('question')
                a = ex.get('answers', [""])
                if hasattr(img, 'filename'):
                    pairs.append({"image": getattr(img, 'filename', ''), "question": q, "answer": a[0] if a else ""})
            if pairs:
                _write_pairs(pairs, "data/vqa/okvqa_eval.jsonl")
                out["okvqa_jsonl"] = "data/vqa/okvqa_eval.jsonl"
        except Exception:
            pass
        # COCO captions min eval JSONL (paths+refs only; predictions filled by your pipeline)
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("coco_captions", "2017", split=f"validation[:{min(limit, 500)}]")
            pairs = []
            for ex in ds:
                img = ex.get('image')
                refs = [c.get('caption','') for c in ex.get('annotations', [])]
                if hasattr(img, 'filename') and refs:
                    pairs.append({"image": getattr(img, 'filename', ''), "references": refs})
            if pairs:
                _write_pairs(pairs, "data/coco/captions_eval.jsonl")
                out["coco_captions_jsonl"] = "data/coco/captions_eval.jsonl"
        except Exception:
            pass
        # MSR-VTT VQA (placeholder if dataset available)
        try:
            # Many MSRVTT loaders need manual auth; keep placeholder path if present
            pass
        except Exception:
            pass
    except Exception:
        pass

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-fetch small real datasets for multiple modalities")
    ap.add_argument("--limit", type=int, default=int(os.getenv("OMNICODER_FETCH_LIMIT", "500")))
    args = ap.parse_args()

    out = autofetch_all(int(args.limit))

    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


