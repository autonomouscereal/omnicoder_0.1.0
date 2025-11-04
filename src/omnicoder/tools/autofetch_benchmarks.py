from __future__ import annotations

"""
autofetch_benchmarks: Download/prepare standard benchmark slices into data/benchmarks/*.jsonl

Uses huggingface datasets to materialize lightweight JSONLs with fields that
our eval scripts can consume. Each dataset is capped by --limit. The goal is to
be fully idempotent and fast to re-run.

Outputs (created if missing):
- data/benchmarks/mmlu.jsonl
- data/benchmarks/arc_easy.jsonl, data/benchmarks/arc_challenge.jsonl
- data/benchmarks/hellaswag.jsonl
- data/benchmarks/truthfulqa_mc.jsonl
- data/benchmarks/winogrande.jsonl
- data/benchmarks/gsm8k.jsonl
- data/benchmarks/mbpp.jsonl
- data/benchmarks/hotpotqa.jsonl
- data/benchmarks/humaneval.jsonl
 - data/benchmarks/vqav2.jsonl (requires COCO 2014 val images)
 - data/benchmarks/okvqa.jsonl (requires COCO 2014 val images)
 - data/benchmarks/coco_captions.jsonl (requires COCO 2017 captions+images)
 - data/benchmarks/msrvtt_vqa.jsonl (video QA questions; video files optional)

Each line follows a common schema for text QA:
  {"prompt": str, "question": str, "choices": [str,...]?, "answer": str}

For code (mbpp, humaneval):
  {"prompt": str, "tests": str}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    try:
        print(f"[autofetch_bench] {msg}", flush=True)
    except Exception:
        pass


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    _ensure_dir(path.parent)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _count_lines(p: Path) -> int:
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def fetch_text_suite(out_root: Path, limit: int) -> Dict[str, str]:
    from datasets import load_dataset  # type: ignore
    paths: Dict[str, str] = {}

    # MMLU (hendrycksTest): combine a few subjects to keep size bounded
    try:
        dst = out_root / "mmlu.jsonl"
        have = _count_lines(dst)
        if have < limit:
            rows: List[Dict[str, Any]] = []
            subjects = [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                "clinical_knowledge", "college_biology", "college_computer_science",
                "college_physics", "high_school_chemistry", "high_school_mathematics",
            ]
            need = limit - have
            for sub in subjects:
                ds = load_dataset("hendrycks_test", sub, split="test")
                for ex in ds:
                    stem = str(ex.get("question", ""))
                    choices = [str(ex.get(k, "")) for k in ("A","B","C","D")]
                    ans = str(ex.get("answer", "")).strip()
                    prompt = stem + "\n" + "\n".join([f"{c}. {choices[i]}" for i, c in enumerate(["A","B","C","D"])])
                    rows.append({"prompt": prompt, "question": stem, "choices": choices, "answer": ans})
                    if len(rows) >= need:
                        break
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["mmlu_jsonl"] = str(dst)
    except Exception as e:
        _log(f"mmlu fetch failed: {e}")

    # ARC (easy, challenge)
    try:
        for split_name, conf in ("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"):
            dst = out_root / f"{split_name}.jsonl"
            have = _count_lines(dst)
            if have < limit:
                ds = load_dataset("ai2_arc", conf, split="test")
                rows = []
                need = limit - have
                for ex in ds:
                    stem = str(ex.get("question", ""))
                    choices = [str(c) for c in (ex.get("choices", {}) or {}).get("text", [])]
                    label = str(ex.get("answerKey", "")).strip()
                    prompt = stem + "\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    rows.append({"prompt": prompt, "question": stem, "choices": choices, "answer": label})
                    if len(rows) >= need:
                        break
                _write_jsonl(dst, rows)
            paths[f"{split_name}_jsonl"] = str(dst)
    except Exception as e:
        _log(f"arc fetch failed: {e}")

    # HellaSwag
    try:
        dst = out_root / "hellaswag.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("hellaswag", split="validation")
            need = limit - have
            rows = []
            for ex in ds:
                ctx = str(ex.get("ctx", ""))
                endings = [str(e) for e in ex.get("endings", [])]
                label = int(ex.get("label", -1))
                prompt = ctx + "\n" + "\n".join([f"{i}. {e}" for i, e in enumerate(endings)])
                rows.append({"prompt": prompt, "question": ctx, "choices": endings, "answer": str(label)})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["hellaswag_jsonl"] = str(dst)
    except Exception as e:
        _log(f"hellaswag fetch failed: {e}")

    # TruthfulQA (multiple choice)
    try:
        dst = out_root / "truthfulqa_mc.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
            need = limit - have
            rows = []
            for ex in ds:
                stem = str(ex.get("question", ""))
                choices = [str(x) for x in (ex.get("mc1_targets", {}) or {}).get("choices", [])]
                labels = (ex.get("mc1_targets", {}) or {}).get("labels", [])
                # Choose first correct label index when available, else 0
                ans_idx = 0
                for i, v in enumerate(labels or []):
                    try:
                        if int(v) == 1:
                            ans_idx = i
                            break
                    except Exception:
                        pass
                prompt = stem + "\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
                rows.append({"prompt": prompt, "question": stem, "choices": choices, "answer": str(ans_idx)})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["truthfulqa_jsonl"] = str(dst)
    except Exception as e:
        _log(f"truthfulqa fetch failed: {e}")

    # Winogrande (validation)
    try:
        dst = out_root / "winogrande.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("winogrande", "winogrande_xl", split="validation")
            need = limit - have
            rows = []
            for ex in ds:
                sent = str(ex.get("sentence", ""))
                opt1 = str(ex.get("option1", "")); opt2 = str(ex.get("option2", ""))
                label = str(ex.get("answer", "1")).strip()
                prompt = sent + "\n1. " + opt1 + "\n2. " + opt2
                rows.append({"prompt": prompt, "question": sent, "choices": [opt1, opt2], "answer": label})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["winogrande_jsonl"] = str(dst)
    except Exception as e:
        _log(f"winogrande fetch failed: {e}")

    # GSM8K
    try:
        dst = out_root / "gsm8k.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("gsm8k", "main", split="test")
            need = limit - have
            rows = []
            for ex in ds:
                q = str(ex.get("question", ""))
                a = str(ex.get("answer", ""))
                rows.append({"prompt": q, "question": q, "answer": a})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["gsm8k_jsonl"] = str(dst)
    except Exception as e:
        _log(f"gsm8k fetch failed: {e}")

    # MBPP
    try:
        dst = out_root / "mbpp.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("mbpp", split="test")
            need = limit - have
            rows = []
            for ex in ds:
                prompt = str(ex.get("text", ""))
                tests = ex.get("test_list") or ex.get("test", "")
                tests_str = "\n".join([str(t) for t in (tests if isinstance(tests, list) else [tests]) if str(t).strip()])
                rows.append({"prompt": prompt, "tests": tests_str})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["mbpp_jsonl"] = str(dst)
    except Exception as e:
        _log(f"mbpp fetch failed: {e}")

    # HotpotQA
    try:
        dst = out_root / "hotpotqa.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("hotpot_qa", "distractor", split="validation")
            need = limit - have
            rows = []
            for ex in ds:
                q = str(ex.get("question", ""))
                a = str(ex.get("answer", ""))
                rows.append({"prompt": q, "question": q, "answer": a})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["hotpot_jsonl"] = str(dst)
    except Exception as e:
        _log(f"hotpotqa fetch failed: {e}")

    # HumanEval
    try:
        dst = out_root / "humaneval.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = load_dataset("openai_humaneval", split="test")
            need = limit - have
            rows = []
            for ex in ds:
                prompt = str(ex.get("prompt", ""))
                tests = str(ex.get("test", ""))
                rows.append({"prompt": prompt, "tests": tests})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["humaneval_jsonl"] = str(dst)
    except Exception as e:
        _log(f"humaneval fetch failed: {e}")

    return paths


def _aria2c(url: str, out: Path) -> bool:
    try:
        import subprocess, shlex
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            _log(f"skip download (exists): {out}")
            return True
        _log(f"downloading: {url} -> {out}")
        cmd = f"aria2c -c -x 8 -s 8 -k 4M -o {shlex.quote(out.name)} --dir {shlex.quote(str(out.parent))} {shlex.quote(url)}"
        rc = subprocess.call(cmd, shell=True)
        ok = rc == 0 and out.exists() and out.stat().st_size > 0
        _log(f"download {'ok' if ok else 'failed'}: {out}")
        return ok
    except Exception as e:
        _log(f"aria2c error: {e}")
        return False


def _ensure_coco2014_val(out_root: Path) -> Dict[str, str]:
    """Ensure COCO 2014 val images exist under data/coco2014/val2014."""
    out: Dict[str, str] = {}
    try:
        root = Path("data/coco2014")
        imgs_zip = root / "val2014.zip"
        imgs_dir = root / "val2014"
        if not imgs_dir.exists():
            url_imgs = "http://images.cocodataset.org/zips/val2014.zip"
            if _aria2c(url_imgs, imgs_zip):
                import zipfile
                _log("extracting COCO val2014.zip ...")
                with zipfile.ZipFile(str(imgs_zip), 'r') as z:
                    z.extractall(str(root))
        if imgs_dir.exists():
            out["coco2014_val_dir"] = str(imgs_dir)
    except Exception as e:
        out["coco2014_error"] = str(e)
    return out


def fetch_vqa_suite(out_root: Path, limit: int) -> Dict[str, str]:
    from datasets import load_dataset  # type: ignore
    paths: Dict[str, str] = {}
    # Ensure COCO2014 val images
    coco_info = _ensure_coco2014_val(out_root)
    coco_val = coco_info.get("coco2014_val_dir", "")
    # VQAv2 (validation)
    try:
        dst = out_root / "vqav2.jsonl"
        have = _count_lines(dst)
        if have < limit:
            # Try multiple builder names commonly used
            ds = None
            for conf in ("v2", "vqa2", "default"):
                try:
                    ds = load_dataset("visual_qa", conf, split="validation")
                    break
                except Exception:
                    try:
                        ds = load_dataset("visualqa", conf, split="validation")
                        break
                    except Exception:
                        continue
            if ds is None:
                raise RuntimeError("could not load vqav2 via datasets")
            rows = []
            need = limit - have
            for ex in ds:
                q = str(ex.get("question", ""))
                ans = ex.get("multiple_choice_answer") or ex.get("answers") or ex.get("answer", "")
                if isinstance(ans, list) and ans:
                    # choose most frequent
                    try:
                        from collections import Counter
                        ans = Counter([str(a.get('answer','')) for a in ans]).most_common(1)[0][0]
                    except Exception:
                        ans = str(ans[0])
                ans = str(ans)
                image_id = int(ex.get("image_id", -1))
                img_name = f"COCO_val2014_{image_id:012d}.jpg"
                img_path = str((Path("data/coco2014/val2014") / img_name).as_posix())
                rows.append({"image": img_path, "question": q, "answer": ans})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["vqav2_jsonl"] = str(dst)
    except Exception as e:
        _log(f"vqav2 fetch failed: {e}")

    # OK-VQA (validation)
    try:
        dst = out_root / "okvqa.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = None
            for name in ("ok_vqa", "okvqa"):
                try:
                    ds = load_dataset(name, split="validation")
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("could not load ok_vqa via datasets")
            rows = []
            need = limit - have
            for ex in ds:
                q = str(ex.get("question", ""))
                a = ex.get("answers") or ex.get("direct_answers") or ex.get("answer", "")
                if isinstance(a, list) and a:
                    a = str(a[0])
                a = str(a)
                img_id = int(ex.get("image_id", -1))
                img_name = f"COCO_val2014_{img_id:012d}.jpg"
                img_path = str((Path("data/coco2014/val2014") / img_name).as_posix())
                rows.append({"image": img_path, "question": q, "answer": a})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        paths["okvqa_jsonl"] = str(dst)
    except Exception as e:
        _log(f"okvqa fetch failed: {e}")

    return paths


def fetch_coco_captions_jsonl(out_root: Path, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        # Reuse COCO 2017 if available
        caps = Path("data/coco2017/annotations/captions_train2017.json")
        imgs_dir = Path("data/coco2017/train2017")
        if caps.exists() and imgs_dir.exists():
            import json as _j
            data = _j.loads(caps.read_text(encoding='utf-8'))
            id_to_file = {int(img.get('id')): str((imgs_dir / img.get('file_name','')).as_posix()) for img in data.get('images', [])}
            dst = out_root / "coco_captions.jsonl"
            have = _count_lines(dst)
            need = max(0, limit - have)
            if need > 0:
                rows = []
                n = 0
                for ann in data.get('annotations', []):
                    img_id = int(ann.get('image_id', -1))
                    imgp = id_to_file.get(img_id, '')
                    txt = str(ann.get('caption', '')).strip()
                    if imgp and txt:
                        rows.append({"image": imgp, "references": [txt]})
                        n += 1
                        if n >= need:
                            break
                _write_jsonl(dst, rows)
            out["coco_captions_jsonl"] = str(dst)
    except Exception as e:
        out["coco_captions_error"] = str(e)
    return out


def fetch_msrvtt_qa(out_root: Path, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        dst = out_root / "msrvtt_vqa.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = None
            for name in ("msrvtt_qa", "msrvtt-qa"):
                try:
                    ds = load_dataset(name, split="test")
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("could not load msrvtt_qa via datasets")
            rows = []
            need = limit - have
            for ex in ds:
                q = str(ex.get('question', ''))
                a = str(ex.get('answer', ''))
                vid = str(ex.get('video_id') or ex.get('video') or '')
                # Expect local videos under data/msrvtt/videos/{video_id}.mp4 if user provides them
                vpath = str((Path('data/msrvtt/videos') / (vid + '.mp4')).as_posix()) if vid else ''
                rows.append({"video": vpath, "video_id": vid, "question": q, "answer": a})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        out["msrvtt_vqa_jsonl"] = str(dst)
    except Exception as e:
        out["msrvtt_vqa_error"] = str(e)
    return out


def fetch_agieval(out_root: Path, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        dst = out_root / "agieval.jsonl"
        have = _count_lines(dst)
        if have < limit:
            subjects = [
                "aqua_rat", "logiqa_en", "math",
                "sat_en", "lsat_ar", "gaokao_en",
            ]
            rows: List[Dict[str, Any]] = []
            need = limit - have
            for sub in subjects:
                try:
                    ds = load_dataset("agieval", sub, split="dev")
                except Exception:
                    continue
                for ex in ds:
                    stem = str(ex.get("question", ""))
                    choices = ex.get("options") or []
                    label = str(ex.get("label", "")).strip()
                    prompt = stem
                    if choices:
                        prompt = stem + "\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
                    rows.append({"prompt": prompt, "question": stem, "choices": choices, "answer": label})
                    if len(rows) >= need:
                        break
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        out["agieval_jsonl"] = str(dst)
    except Exception as e:
        out["agieval_error"] = str(e)
    return out


def fetch_bbh(out_root: Path, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        dst = out_root / "bbh.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = None
            for name in ("bigbench-hard", "bbh"):
                try:
                    ds = load_dataset(name, split="test")
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("could not load bigbench-hard")
            rows = []
            need = limit - have
            for ex in ds:
                stem = str(ex.get("input", ""))
                ans = str(ex.get("target", ""))
                rows.append({"prompt": stem, "question": stem, "answer": ans})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        out["bbh_jsonl"] = str(dst)
    except Exception as e:
        out["bbh_error"] = str(e)
    return out


def fetch_swebench_meta(out_root: Path, limit: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore
        dst = out_root / "swebench.jsonl"
        have = _count_lines(dst)
        if have < limit:
            ds = None
            for name in ("princeton-nlp/SWE-bench_Lite", "swe-bench", "swebench"):
                try:
                    ds = load_dataset(name, split="test")
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError("could not load swebench meta")
            rows = []
            need = limit - have
            for ex in ds:
                rid = str(ex.get("instance_id") or ex.get("id") or "")
                repo = str(ex.get("repo") or ex.get("repo_name") or "")
                base = str(ex.get("base_commit") or ex.get("base_sha") or "")
                tests = ex.get("test_commands") or ex.get("test") or []
                tests = tests if isinstance(tests, list) else [tests]
                rows.append({"instance_id": rid, "repo": repo, "base": base, "tests": tests})
                if len(rows) >= need:
                    break
            _write_jsonl(dst, rows)
        out["swebench_jsonl"] = str(dst)
    except Exception as e:
        out["swebench_error"] = str(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Autofetch standard benchmark datasets")
    ap.add_argument("--out_root", type=str, default="data/benchmarks")
    ap.add_argument("--limit", type=int, default=int(os.getenv("OMNICODER_BENCH_FETCH_LIMIT", "200")))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {}

    try:
        out.update(fetch_text_suite(out_root, args.limit))
    except Exception as e:
        out["error_text_suite"] = str(e)

    try:
        out.update(fetch_vqa_suite(out_root, args.limit))
    except Exception as e:
        out["error_vqa_suite"] = str(e)

    # Optional: extended suites (best-effort). We ignore failures and proceed.
    try:
        out.update(fetch_coco_captions_jsonl(out_root, args.limit))
    except Exception as e:
        out["error_coco_captions"] = str(e)
    try:
        out.update(fetch_msrvtt_qa(out_root, args.limit))
    except Exception as e:
        out["error_msrvtt_vqa"] = str(e)

    try:
        out.update(fetch_agieval(out_root, args.limit))
    except Exception as e:
        out["error_agieval"] = str(e)
    try:
        out.update(fetch_bbh(out_root, args.limit))
    except Exception as e:
        out["error_bbh"] = str(e)
    try:
        out.update(fetch_swebench_meta(out_root, args.limit))
    except Exception as e:
        out["error_swebench"] = str(e)

    try:
        out.update(fetch_coco_captions_jsonl(out_root, args.limit))
    except Exception as e:
        out["error_coco_captions"] = str(e)

    try:
        out.update(fetch_msrvtt_qa(out_root, args.limit))
    except Exception as e:
        out["error_msrvtt_vqa"] = str(e)

    # Write manifest
    (out_root / "MANIFEST.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


