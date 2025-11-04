# Teacher models (by domain/preset)

Pick strong, readily-available teachers with good license posture. Place weights under `/models/hf` to avoid re-downloads.

- Text (general)
  - Llama 3.1 8B/70B (where licensed), Mistral 7B Instruct, Qwen2.5 7B/14B, Phi-3-mini (lightweight), Gemma 2 9B
- Code
  - StarCoder2-3B/7B, Qwen2.5-Coder 7B, CodeLlama 7B/13B
- Vision-language (image-text)
  - OpenCLIP ViT-B/32, ViT-L/14, SigLIP; BLIP-2/IDEFICS for captioning supervision
- ASR/TTS
  - ASR: faster-whisper-medium/large-v3
  - TTS: Coqui XTTS v2, Piper voices (on-device)
- Image generation
  - Stable Diffusion v1.5 / SDXL-lite variants; SD Turbo for drafts
- Video generation
  - Modelscope T2V / zeroscope-like lite variants (research), I2V via SD + interpolation

Profiles
- Edit `profiles/teachers.json` to set defaults per preset (`mobile_4gb`, `mobile_2gb`, `draft_2b`, …).
- Or override via env: `OMNICODER_TEACHER`, `OMNICODER_KD_{TEXT,CODE,VL,ASR,TTS}_TEACHERS`.

Notes
- For KD on a single 24 GB GPU, prefer bf16/fp16 with `--teacher_device_map auto` and low batch size.
- Draft acceptance improves with stronger drafts (2–3B). Use `OMNICODER_DRAFT_PRESET=draft_2b|draft_3b`.
