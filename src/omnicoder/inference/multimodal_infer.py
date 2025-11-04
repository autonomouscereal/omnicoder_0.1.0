import argparse
import torch
import numpy as np

from omnicoder.inference.generate import (
    generate,
    build_mobile_model,
    build_mobile_model_by_name,
    maybe_load_checkpoint,
    prime_kv_with_features,
    continue_generate_from_primed,
)
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.config import MobilePreset
from omnicoder.modeling.multimodal.image_pipeline import ImageGenPipeline
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.modeling.multimodal.video_pipeline import VideoGenPipeline
from omnicoder.modeling.multimodal.asr import ASRAdapter
from omnicoder.modeling.multimodal.tts import TTSAdapter
from omnicoder.modeling.multimodal.audio_tokenizer import AudioTokenizer
from omnicoder.modeling.multimodal.audio_vocoder import HiFiGANVocoder
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.aligner import PreAligner, TextEmbedder


def _run_text(prompt: str, max_new_tokens: int, mobile_preset: str, ckpt: str, device: str) -> None:
    tokenizer = get_text_tokenizer(prefer_hf=True)

    if mobile_preset in ('mobile_4gb', 'mobile_2gb'):
        model = build_mobile_model_by_name(mobile_preset)
    else:
        from omnicoder.modeling.transformer_moe import OmniTransformer
        model = OmniTransformer()
    maybe_load_checkpoint(model, ckpt)
    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out_ids = generate(model, input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(out_ids[0].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["text", "image", "video", "audio"], required=True)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--load_latent_heads", type=str, default="", help="Optional path to a heads-only state dict (image/audio latent heads)")
    ap.add_argument("--device", type=str, default="cpu")
    # Optional multimodal inputs for text task (fused understanding)
    ap.add_argument("--image_input", type=str, default="", help="Path to an image to fuse with the prompt for text generation")
    ap.add_argument("--video_input", type=str, default="", help="Path to a video file or a directory of frames for fused text generation")
    # Image-specific
    ap.add_argument("--image_backend", type=str, default="diffusers", choices=["diffusers", "onnx"], help="Image generation backend")
    ap.add_argument("--sd_model", type=str, default="", help="HF model id for Stable Diffusion, e.g., runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_local_path", type=str, default="", help="Local path to a diffusers pipeline directory")
    ap.add_argument("--onnx_sd_dir", type=str, default="", help="Path to ONNX-exported SD pipeline (Optimum export); enables ORT callable injection")
    ap.add_argument("--onnx_provider", type=str, default="CPUExecutionProvider", help="ONNX Runtime provider (e.g., CPUExecutionProvider, DmlExecutionProvider, NNAPIExecutionProvider)")
    ap.add_argument("--provider_profile", type=str, default="", help="Optional provider profile JSON for ONNX image backend (overrides --onnx_provider)")
    ap.add_argument("--image_steps", type=int, default=20, help="Diffusion steps")
    ap.add_argument("--image_width", type=int, default=512)
    ap.add_argument("--image_height", type=int, default=512)
    ap.add_argument("--image_out", type=str, default="weights/image_out.png")
    ap.add_argument("--image_refiner_steps", type=int, default=0, help="If >0, run tiny image refiner for N steps before saving")
    # Optional cross-modal verifier args
    ap.add_argument("--cm_verifier", action="store_true", default=(os.environ.get("OMNICODER_CM_VERIFIER", "0") == "1"))
    ap.add_argument("--cm_threshold", type=float, default=float(os.environ.get("OMNICODER_CM_THRESHOLD", "0.6")))
    # Video-specific
    ap.add_argument("--video_backend", type=str, default="diffusers", choices=["diffusers"], help="Video generation backend")
    ap.add_argument("--video_model", type=str, default="", help="HF id for text-to-video/diffusion pipeline")
    ap.add_argument("--video_local_path", type=str, default="", help="Local path to video pipeline")
    ap.add_argument("--video_frames", type=int, default=24)
    ap.add_argument("--video_steps", type=int, default=25)
    ap.add_argument("--video_width", type=int, default=512)
    ap.add_argument("--video_height", type=int, default=320)
    ap.add_argument("--video_out", type=str, default="weights/video_out.mp4")
    ap.add_argument("--video_seed_image", type=str, default="", help="Seed image for image-to-video backends")
    ap.add_argument("--onnx_video_dir", type=str, default="", help="Path to ONNX-exported i2v pipeline (expects generator.onnx)")
    ap.add_argument("--onnx_video_provider", type=str, default="CPUExecutionProvider", help="ORT provider for i2v (e.g., NNAPIExecutionProvider)")
    ap.add_argument("--onnx_video_provider_profile", type=str, default="", help="Provider profile JSON for i2v callable")
    # Temporal consistency filter controls
    ap.add_argument("--temporal_filter", action="store_true", default=(os.environ.get("OMNICODER_VIDEO_TEMPORAL_FILTER", "1") == "1"))
    ap.add_argument("--temporal_alpha", type=float, default=float(os.environ.get("OMNICODER_VIDEO_TEMPORAL_ALPHA", "0.7")))
    ap.add_argument("--temporal_passes", type=int, default=int(os.environ.get("OMNICODER_VIDEO_TEMPORAL_PASSES", "1")))
    # Audio-specific
    ap.add_argument("--asr_input", type=str, default="", help="Path to audio file for ASR")
    ap.add_argument("--asr_model_size", type=str, default="small")
    ap.add_argument("--tts_text", type=str, default="", help="Text to synthesize with TTS")
    ap.add_argument("--tts_out", type=str, default="weights/tts_out.wav")
    ap.add_argument("--tts_model", type=str, default="tts_models/en/ljspeech/tacotron2-DDC")
    # Audio codec (EnCodec) roundtrip
    ap.add_argument("--audio_tokenize_in", type=str, default="", help="Path to WAV/MP3 to tokenize with EnCodec and optionally reconstruct")
    ap.add_argument("--audio_reconstruct_out", type=str, default="", help="If set, reconstruct audio from codes to this WAV path")
    ap.add_argument("--audio_map_vocab", action="store_true", help="Map audio codes to unified vocab and print range sample")
    # Audio VQ-VAE quick roundtrip (in-memory)
    ap.add_argument("--audio_vqvae_roundtrip", action="store_true", help="Perform a small Audio VQ-VAE encode/decode roundtrip on a random waveform segment and report MSE")
    # Vocoder (HiFi-GAN) direct mel->wav
    ap.add_argument("--mel_npy", type=str, default="", help="Path to a mel-spectrogram .npy (n_mels,T) to vocode with HiFi-GAN")
    ap.add_argument("--vocoder_backend", type=str, default="auto", choices=["auto","coqui_hifigan","onnx","torch"], help="HiFi-GAN backend selection")
    ap.add_argument("--vocoder_model", type=str, default="", help="Path to ONNX/TorchScript HiFi-GAN model (if using onnx/torch backends)")
    args = ap.parse_args()

    # Optional: load unified vocab sidecar for mapping checks or offsets
    vocab_layout = None
    try:
        from omnicoder.modeling.multimodal.vocab_map import VocabSidecar
        import os
        sidecar = os.getenv("OMNICODER_VOCAB_SIDECAR", "weights/unified_vocab_map.json")
        from pathlib import Path as _P
        if _P(sidecar).exists():
            vocab_layout = VocabSidecar.load(sidecar).as_layout()
    except Exception:
        pass

    if args.task == "text":
        # If an image is provided, run fused image+text → text generation using KV priming
        if args.image_input:
            from PIL import Image  # type: ignore
            import numpy as np
            tokenizer = get_text_tokenizer(prefer_hf=True)
            if args.mobile_preset in ("mobile_4gb", "mobile_2gb"):
                model = build_mobile_model_by_name(args.mobile_preset)
            else:
                model = OmniTransformer()
            maybe_load_checkpoint(model, args.ckpt)
            if args.load_latent_heads:
                try:
                    heads_sd = torch.load(args.load_latent_heads, map_location='cpu')
                    model.load_state_dict(heads_sd, strict=False)
                    print(f"Loaded latent heads from {args.load_latent_heads}")
                except Exception as e:
                    print(f"[heads] could not load latent heads: {e}")
            model.to(args.device)

            # Prepare fused features
            img = Image.open(args.image_input).convert("RGB").resize((224, 224))
            img_np = np.array(img).astype("float32") / 255.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
            composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
            fused = composer.fuse_text_image(
                model_with_embed=model,
                input_ids=torch.tensor([[tokenizer.encode(args.prompt or "Describe the image")]], dtype=torch.long).squeeze(0),
                image_bchw=img_t,
            )
            # Prime KV and continue decoding from BOS
            past_kv, _ = prime_kv_with_features(model, fused)
            bos_id = 1 if hasattr(tokenizer, "bos_token_id") else 2  # fallback
            out_ids = continue_generate_from_primed(
                model,
                past_kv=past_kv,
                start_token_id=bos_id,
                max_new_tokens=args.max_new_tokens,
            )
            print(tokenizer.decode(out_ids[0].tolist()))
        elif args.video_input:
            # Fused video+text → text
            import os
            import numpy as np
            tokenizer = get_text_tokenizer(prefer_hf=True)
            if args.mobile_preset in ("mobile_4gb", "mobile_2gb"):
                model = build_mobile_model_by_name(args.mobile_preset)
            else:
                model = OmniTransformer()
            maybe_load_checkpoint(model, args.ckpt)
            model.to(args.device)

            # Load frames from a video file or directory of images
            path = args.video_input
            frames_np = []
            target_wh = (224, 224)
            def _resize_np(arr):
                try:
                    import cv2  # type: ignore
                    return cv2.resize(arr, target_wh, interpolation=cv2.INTER_AREA)
                except Exception:
                    from PIL import Image  # type: ignore
                    return np.array(Image.fromarray(arr).resize(target_wh))

            if os.path.isdir(path):
                files = sorted([f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg"))])
                for f in files:
                    from PIL import Image  # type: ignore
                    arr = np.array(Image.open(os.path.join(path, f)).convert("RGB"))
                    frames_np.append(_resize_np(arr))
            else:
                try:
                    import cv2  # type: ignore
                    cap = cv2.VideoCapture(path)
                    ok = True
                    while ok:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_np.append(_resize_np(frame))
                    cap.release()
                except Exception:
                    pass

            if not frames_np:
                print("Could not read video frames from --video_input")
                return
            # Subsample up to 16 evenly
            max_frames = 16
            t = len(frames_np)
            if t > max_frames:
                idx = np.linspace(0, t - 1, num=max_frames).round().astype(int)
                frames_np = [frames_np[i] for i in idx]

            frames = np.stack(frames_np, axis=0).astype("float32") / 255.0  # (T,H,W,C)
            video_btchw = torch.from_numpy(frames).permute(0,3,1,2).unsqueeze(0)  # (1,T,3,H,W)

            composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
            fused = composer.fuse_text_video(
                model_with_embed=model,
                input_ids=torch.tensor([[tokenizer.encode(args.prompt or "Describe the video")]], dtype=torch.long).squeeze(0),
                video_btchw=video_btchw,
                max_frames=16,
            )
            past_kv, _ = prime_kv_with_features(model, fused)
            bos_id = 1 if hasattr(tokenizer, "bos_token_id") else 2
            out_ids = continue_generate_from_primed(
                model,
                past_kv=past_kv,
                start_token_id=bos_id,
                max_new_tokens=args.max_new_tokens,
            )
            print(tokenizer.decode(out_ids[0].tolist()))
        else:
            _run_text(args.prompt, args.max_new_tokens, args.mobile_preset, args.ckpt, args.device)
    elif args.task == "image":
        # Optional: condition the image generator using the LLM hidden state
        tokenizer = get_text_tokenizer(prefer_hf=True)
        if args.mobile_preset in ('mobile_4gb', 'mobile_2gb'):
            model = build_mobile_model_by_name(args.mobile_preset)
        else:
            model = OmniTransformer()
        maybe_load_checkpoint(model, args.ckpt)
        if args.load_latent_heads:
            try:
                heads_sd = torch.load(args.load_latent_heads, map_location='cpu')
                model.load_state_dict(heads_sd, strict=False)
            except Exception as e:
                print(f"[heads] could not load latent heads: {e}")
        model.to(args.device)
        inp = torch.tensor([tokenizer.encode(args.prompt or "Describe image")], dtype=torch.long)
        outputs = model(inp, return_hidden=True)
        hidden = None
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                _, hidden = outputs
            elif len(outputs) == 3:
                hidden = outputs[2]  # type: ignore[index]
        # Choose dtype based on device
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        # Initialize image pipeline
        backend = args.image_backend
        hf_id = args.sd_model if args.sd_model else None
        local_path = args.sd_local_path if args.sd_local_path else None
        pipe = ImageGenPipeline(backend=backend, device=args.device, dtype=dtype, hf_id=hf_id, local_path=local_path)
        # Optional: if an ONNX pipeline dir is provided, inject ORT callable and skip diffusers loader
        if backend == "onnx" and args.onnx_sd_dir:
            try:
                import os, json
                from omnicoder.inference.runtimes.onnx_image_decode import ORTSDCallable
                prof_path = args.provider_profile or os.getenv("OMNICODER_IMAGE_PROVIDER_PROFILE", os.getenv("OMNICODER_PROVIDER_PROFILE", ""))
                provider = args.onnx_provider
                provider_options = None
                if prof_path:
                    try:
                        data = json.loads(open(prof_path, "r", encoding="utf-8").read())
                        provider = str(data.get("provider", provider))
                        provider_options = data.get("provider_options", None)
                    except Exception:
                        pass
                ort_callable = ORTSDCallable(args.onnx_sd_dir, provider=provider, provider_options=provider_options)
                pipe.load_backend(pipe=ort_callable)
                ok = True
            except Exception as e:
                print(f"Failed to initialize ORT SD callable: {e}")
                return
        else:
            ok = pipe.ensure_loaded()
            if not ok:
                print("Install diffusers (or provide --onnx_sd_dir) and provide a local model path or HF id to enable image generation.")
                return
        # Build a normalized text embedding using PreAligner + a tiny TextEmbedder when gating is requested
        text_embed = None
        if bool(args.cm_verifier):
            try:
                ed = 256
                # Best-effort: reuse pre-align checkpoint if present
                from pathlib import Path as _P
                ck = None
                ck_path = _P("weights/pre_align.pt")
                if ck_path.exists():
                    ck = torch.load(str(ck_path), map_location="cpu")
                    ed = int(ck.get("embed_dim", ed))
                tok = get_text_tokenizer(prefer_hf=True)
                vocab_size = int(getattr(tok, "vocab_size", 32000))
                txt_ids = torch.tensor([[tok.encode(args.prompt or "A scenic landscape")]], dtype=torch.long).squeeze(0)
                txt_ids = txt_ids[: max(1, min(64, txt_ids.numel()))].unsqueeze(0)
                te = TextEmbedder(vocab_size=vocab_size, embed_dim=ed).eval()
                with torch.no_grad():
                    tvec = te(txt_ids)
                pal = PreAligner(embed_dim=ed, text_dim=ed, image_dim=768).eval()
                if ck is not None and isinstance(ck, dict) and "aligner" in ck:
                    try:
                        pal.load_state_dict(ck["aligner"])  # type: ignore[index]
                    except Exception:
                        pass
                with torch.no_grad():
                    emb = pal(text=tvec)
                text_embed = emb.get("text", None)
            except Exception:
                text_embed = None

        out = pipe.generate(
            args.prompt or "A scenic landscape",
            conditioning=hidden,
            steps=int(args.image_steps),
            size=(int(args.image_width), int(args.image_height)),
            out_path=args.image_out,
            refiner_steps=int(args.image_refiner_steps),
            cm_verifier=bool(args.cm_verifier),
            cm_threshold=float(args.cm_threshold),
            text_embed=text_embed,
        )
        print(f"Saved image to: {out}")
    elif args.task == "video":
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        backend = args.video_backend
        hf_id = args.video_model or None
        local_path = args.video_local_path or None
        vpipe = VideoGenPipeline(backend=backend, device=args.device, dtype=dtype, hf_id=hf_id, local_path=local_path)
        ok = vpipe.ensure_loaded()
        if not ok:
            print("Install diffusers and provide a local/HF video pipeline (e.g., TextToVideoSDPipeline).")
            return
        # Optional cross-modal verifier: prepare a text embedding
        text_embed = None
        try:
            if bool(args.cm_verifier):
                ed = 256
                from pathlib import Path as _P
                ck = None
                ck_path = _P("weights/pre_align.pt")
                if ck_path.exists():
                    ck = torch.load(str(ck_path), map_location="cpu")
                    ed = int(ck.get("embed_dim", ed))
                tok = get_text_tokenizer(prefer_hf=True)
                vocab_size = int(getattr(tok, "vocab_size", 32000))
                txt_ids = torch.tensor([[tok.encode(args.prompt or "A scenic landscape in motion")]], dtype=torch.long).squeeze(0)
                txt_ids = txt_ids[: max(1, min(64, txt_ids.numel()))].unsqueeze(0)
                te = TextEmbedder(vocab_size=vocab_size, embed_dim=ed).eval()
                with torch.no_grad():
                    tvec = te(txt_ids)
                pal = PreAligner(embed_dim=ed, text_dim=ed, image_dim=768).eval()
                if ck is not None and isinstance(ck, dict) and "aligner" in ck:
                    try:
                        pal.load_state_dict(ck["aligner"])  # type: ignore[index]
                    except Exception:
                        pass
                with torch.no_grad():
                    emb = pal(text=tvec)
                text_embed = emb.get("text", None)
        except Exception:
            text_embed = None

        out = vpipe.generate(
            args.prompt or "A scenic landscape in motion",
            steps=int(args.video_steps),
            size=(int(args.video_width), int(args.video_height)),
            num_frames=int(args.video_frames),
            out_path=args.video_out,
            seed_image=(args.video_seed_image or None),
            temporal_filter=bool(args.temporal_filter),
            temporal_alpha=float(args.temporal_alpha),
            temporal_passes=int(args.temporal_passes),
            onnx_video_dir=(args.onnx_video_dir or None),
            onnx_provider=args.onnx_video_provider,
            onnx_provider_profile=(args.onnx_video_provider_profile or None),
            cm_verifier=bool(args.cm_verifier),
            cm_threshold=float(args.cm_threshold),
            text_embed=text_embed,
        )
        if out is None:
            print("Video pipeline loaded but generation could not proceed (missing T2V backend or seed image for I2V).")
            return
        print(f"Saved video to: {out}")
    else:
        did_any = False
        # ASR
        if args.asr_input:
            asr = ASRAdapter(model_size=args.asr_model_size)
            text = asr.transcribe(args.asr_input)
            print(text or "")
            did_any = True
        # TTS
        if args.tts_text:
            tts = TTSAdapter(model_name=args.tts_model)
            out = tts.tts(args.tts_text, out_path=args.tts_out)
            if out:
                print(f"Saved TTS audio to: {out}")
            else:
                print("TTS backend not available.")
            did_any = True
        # Audio codec roundtrip
        if args.audio_tokenize_in:
            try:
                import soundfile as sf  # type: ignore
                import numpy as np
                wav, sr = sf.read(args.audio_tokenize_in)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                # Normalize to [-1,1]
                wav = wav.astype("float32")
                mx = max(1e-6, float(np.abs(wav).max()))
                wav = wav / mx
                tok = AudioTokenizer(sample_rate=sr)
                codes = tok.encode(wav)
                print(f"Audio tokenized into {len(codes)} codebooks with shapes {[c.shape for c in codes]}")
                if args.audio_map_vocab:
                    try:
                        from omnicoder.modeling.multimodal.vocab_map import map_audio_tokens
                        from omnicoder.config import MultiModalConfig
                        mmc = MultiModalConfig()
                        mapped = map_audio_tokens(codes[0].tolist(), mmc)
                        print(f"Mapped first codebook into unified vocab range [{mmc.audio_vocab_start}, {mmc.audio_vocab_start + mmc.audio_codebook_size - 1}], sample: {mapped[:8]}")
                    except Exception as _e:
                        print(f"Audio vocab mapping failed: {_e}")
                if args.audio_reconstruct_out:
                    rec = tok.decode(codes)
                    if rec is not None:
                        sf.write(args.audio_reconstruct_out, rec.squeeze(), sr)
                        print(f"Reconstructed audio saved to: {args.audio_reconstruct_out}")
                did_any = True
            except Exception as e:
                print(f"Audio tokenize/reconstruct failed: {e}")
        # Audio VQ-VAE roundtrip demo
        if args.audio_vqvae_roundtrip:
            try:
                import numpy as _np
                from omnicoder.modeling.multimodal.audio_vqvae import AudioVQVAE
                seg = _np.random.default_rng(0).standard_normal(32768).astype("float32")
                seg = seg / max(1e-6, float(_np.abs(seg).max()))
                m = AudioVQVAE().eval()
                with torch.no_grad():
                    rec_l, com_l, ppx, xr, idx = m(torch.from_numpy(seg[None, None, :]))
                print(f"Audio VQ-VAE roundtrip rec_mse={rec_l.item():.6f} com={com_l.item():.6f} ppx={float(ppx):.2f}")
                did_any = True
            except Exception as e:
                print(f"Audio VQ-VAE roundtrip failed: {e}")
        # Mel -> waveform vocoding (HiFi-GAN)
        if args.mel_npy:
            try:
                mel = np.load(args.mel_npy).astype("float32")
                voc = HiFiGANVocoder(backend=args.vocoder_backend, onnx_model_path=(args.vocoder_model or None))
                wav = voc.vocode(mel, out_path=args.tts_out)
                if wav is not None:
                    print(f"Saved vocoded audio to: {args.tts_out}")
                else:
                    print("Vocoder failed; ensure backend/model is available.")
                did_any = True
            except Exception as e:
                print(f"Vocoder failed: {e}")
        if not did_any:
            print("No audio actions requested. Use --asr_input, --tts_text, or --audio_tokenize_in.")


if __name__ == "__main__":
    main()
