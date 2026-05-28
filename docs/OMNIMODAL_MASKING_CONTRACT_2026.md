# Omnimodal Masking Contract 2026

This note defines what the active loss mask does and what it does not do.

## Core Rule

OmniCoder uses one causal decoder trunk for text, code, tools, math, image,
video, OCR, TTS, audio, and music token records. The loss mask is only a
supervision mask. It is not a modality router, adapter switch, attention
partition, or separate-head selector.

All prompt, trace, tool, and media input tokens remain in the same token
sequence. They are embedded by the same embedding table, processed by the same
attention/MLP layers, and condition later answer or artifact tokens through the
same causal self-attention path. Target loss is then applied only to assistant
answer tokens and media artifact/output tokens.

## Why Prompt/Input Tokens Are Masked

Prompt and context tokens are masked from cross-entropy so the model is not
rewarded for copying the user prompt, reconstructing benchmark input text, or
learning arbitrary file-path echoes. Their hidden states still matter: the
unmasked assistant and artifact tokens are predicted from those earlier context
tokens, so gradients update the shared weights that processed the complete
sequence.

The active dataset path therefore uses:

- `labels = -100` for user, system, developer, tool-observation, benchmark
  prompt, and media input/context tokens.
- `labels = token_id` for assistant answer tokens, tool-call outputs, and
  image/video/music/TTS/OCR artifact tokens.
- The same token IDs, embedding table, trunk layers, and output head for every
  modality.

## Media Input And Output

Media inputs must be serialized as ledger tokens or canonical media token
records inside the prompt/context side of the sequence. The training path now
preserves `input_json` media payloads such as `image_tokens`, `video_tokens`,
`audio_tokens`, `music_tokens`, `tts_tokens`, `ocr_image_tokens`,
`input_media_tokens`, and reference media metadata as unmasked context. They
are not dropped just because the prompt has a plain text field.

Media outputs are target tokens. A record that asks for an image, video, music,
TTS, audio, or OCR output trains the assistant side to emit a visible route
prefix plus canonical artifact JSON/tokens, for example:

```text
assistant: image | {"output_modality":"image","artifact_tokens":"<image_begin> ... <image_end>"}
```

The generated route/artifact stream is consumed by edge decoders and artifact
renderers after the shared trunk emits it. Those edge decoders are runtime
interfaces, not extra reasoning adapters inside the model.

Minimal supervised media bridge row:

```json
{
  "input_json": {
    "prompt": "Describe and edit the source image.",
    "input_modality": "image",
    "image_tokens": "<image_begin> ... <image_end>"
  },
  "target_json": {
    "output_modality": "image",
    "artifact_tokens": "<image_begin> edited ... <image_end>",
    "caption": "Edited source image"
  }
}
```

In this row, `prompt`, `input_modality`, and `image_tokens` are context
(`labels = -100`). The assistant caption, `image |` route prefix, and artifact
tokens are supervised targets (`labels = token_id`).

## Cross-Modality Transfer Requirement

The mask alone does not guarantee deep omnimodal understanding. Cross-modality
transfer comes from the mixture design:

- Mixed batches update the same trunk and head from text, code, tools, math,
  images, video, audio, TTS, music, and OCR rows.
- Bridge rows connect modalities directly: image to OCR/text, audio to text,
  video to question answering, text to image/video/audio/music, image+text to
  edits, tool traces to media artifact outputs, and code/math tasks grounded in
  media contexts.
- The curriculum must keep bridge tasks present at every phase so modality
  manifolds do not drift into isolated islands.

For full training, the data manifest should report per-modality rows and
cross-modal bridge counts. A run that has media output rows but no media input
tokens, or vice versa, is not considered an omnimodal-ready run.

The expected manifest proof keys are:

- `source_counts`: row count by source family, for example Qwen text/code/tool,
  Qwen image generate/edit, LTX video, ACE/music/TTS, OCR, traces, math, and
  benchmark-derived heldout diagnostics.
- `modality_counts`: row count by primary training modality: `text`, `code`,
  `tool`, `math`, `image`, `video`, `music`, `tts`, `audio`, `ocr`, and
  `bridge`.
- `bridge_counts`: direct cross-modal row counts such as `image_to_text`,
  `text_to_image`, `image_text_to_image_edit`, `text_to_video`,
  `video_to_text`, `text_to_music`, `text_to_tts`, `audio_to_text`, and
  `ocr_image_to_text`.
- `target_coverage`: supervised target-token presence for assistant answers,
  tool outputs, route prefixes, and media artifact tokens.
- `context_coverage`: masked input-token presence for media/context tokens so
  the run proves inputs were kept in the shared sequence rather than dropped.

## Validation Gates

Before a full run is trusted, diagnostics must show:

- Assistant answer targets are covered.
- Media artifact targets are covered for image, video, music, TTS/audio, and
  OCR.
- Media input tokens are present as masked context, not discarded.
- Greedy decode from scratch can learn the route/artifact format on a tiny
  controlled set.
- Heldout sample loss reports non-null loss/perplexity across text, code, tool,
  math, image, video, music, TTS/audio, OCR, and bridge tasks.
- Short-context generation is sane before the 8K -> 32K -> 128K -> 262K ->
  524K -> 1M ladder starts.
