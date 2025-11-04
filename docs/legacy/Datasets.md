# Datasets (curated starting points)

This list groups high-ROI public datasets by modality. Verify licenses and terms for your use before training.

- Text
  - The Pile, SlimPajama, C4, Dolma
  - Instruction/alignment: UltraChat, OpenOrca, FLAN mixtures, Alpaca/ShareGPT (filtered)
  - Long-context: BookCorpus (CC), PG-19, arXiv (via LAION), LongBench-style corpora
- Code
  - The Stack v2 (filtered), StarCoderData, CodeParrot, CodeAlpaca (inst.), HumanEval/MBPP (eval)
- Vision (image)
  - LAION-2B/400M (filtered), COCO, OpenImages, ImageNet-1k (non-commercial), SAM/SA-1B (masks; license)
  - Grounding: VisualGenome, RefCOCO/COCO-Captions, OWL/OVD annotations
- Vision-Language (VL/VQA)
  - LAION-COCO Captions, CC3M/CC12M, COYO, WebLI (where permitted), VQAv2/GQA/OK-VQA/TextCaps
- Video
  - WebVid-10M, HD-VILA, HowTo100M (license), Something-Something V2, UCF-101/Kinetics (eval)
- Audio (ASR/TTS/music)
  - ASR: Common Voice, LibriSpeech/LibriLight, GigaSpeech, VoxPopuli
  - TTS: LJSpeech, M-AILABS (license), CMU Arctic; multi-speaker corpora (check rights)
  - General: AudioSet (weak labels), VGGSound
  - Music: MAESTRO, MusicNet (eval/analysis)
- Multimodal alignment (Image/Text/Audio/Video)
  - CLIP-style pairs from LAION/COCO; AudioCaps/Clotho; VATEX (video-text)

Synthetic/augmentation (optional)
- Self-instruct for text/code; CLIP-guided captioning for images; Whisper transcription for raw audio; TTS back-translation for ASR.
- Noise/drop: random token jitter/drop, span masking, distractor insertion (see env knobs in `training/data/datamodule.py`).

Data engine hooks
- See `src/omnicoder/training/data/engine.py` and `profiles/datasets.json` to wire local corpora/caches. Persist large artifacts under `/models`.
