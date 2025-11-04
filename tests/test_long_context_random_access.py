import os
from pathlib import Path
import torch


def test_windowed_decode_with_landmarks_runs(tmp_path, monkeypatch):
	# Enable landmarks by default for this run and set a target context to derive defaults
	monkeypatch.setenv('OMNICODER_USE_LANDMARKS', '1')
	monkeypatch.setenv('OMNICODER_TARGET_CTX', '32768')
	from omnicoder.inference.generate import build_mobile_model_by_name, generate
	from omnicoder.training.simple_tokenizer import get_text_tokenizer

	tok = get_text_tokenizer(prefer_hf=True)
	# Create a long synthetic prompt
	long_text = ("This is a long document. " * 512) + " Question: what is repeated?"
	input_ids = torch.tensor([tok.encode(long_text)], dtype=torch.long)

	model = build_mobile_model_by_name('mobile_2gb', mem_slots=4)
	model.eval()

	# Force small window to trigger memory priming + random-access landmarks
	out_ids = generate(
		model,
		input_ids,
		max_new_tokens=8,
		temperature=0.8,
		top_k=20,
		top_p=0.9,
		kvq='none',
		window_size=256,
		adaptive_top_k_min=1,
		adaptive_top_k_max=2,
		adaptive_conf_floor=0.3,
		adaptive_layer_ramp=False,
	)
	assert out_ids.shape[1] >= input_ids.shape[1]


