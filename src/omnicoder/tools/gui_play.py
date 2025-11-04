import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

import torch

from omnicoder.inference.generate import build_mobile_model_by_name, generate
from omnicoder.training.simple_tokenizer import get_text_tokenizer


class PlayGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("OmniCoder Play")
        self.device = tk.StringVar(value="cpu")
        self.preset = tk.StringVar(value="mobile_4gb")
        self.prompt = tk.StringVar(value="Hello, OmniCoder!")
        self.tokens = tk.IntVar(value=64)

        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(frm, text="Device:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.device, values=["cpu","cuda"], width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="Preset:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(frm, textvariable=self.preset, values=["mobile_4gb","mobile_2gb"], width=12).grid(row=0, column=3, sticky="w")
        ttk.Label(frm, text="# Tokens:").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm, textvariable=self.tokens, width=6).grid(row=0, column=5, sticky="w")

        ttk.Label(frm, text="Prompt:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.prompt, width=80).grid(row=1, column=1, columnspan=5, sticky="we")

        self.run_btn = ttk.Button(frm, text="Generate", command=self._on_generate)
        self.run_btn.grid(row=2, column=0, sticky="w")

        self.out = scrolledtext.ScrolledText(frm, width=96, height=18, wrap=tk.WORD)
        self.out.grid(row=3, column=0, columnspan=6, sticky="nsew")
        for c in range(6):
            frm.columnconfigure(c, weight=1)
        frm.rowconfigure(3, weight=1)

    def _on_generate(self) -> None:
        self.run_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._run_model, daemon=True).start()

    def _run_model(self) -> None:
        try:
            tok = get_text_tokenizer(prefer_hf=True)
            model = build_mobile_model_by_name(self.preset.get())
            model.to(self.device.get())
            input_ids = torch.tensor([tok.encode(self.prompt.get())], dtype=torch.long)
            out_ids = generate(model, input_ids, max_new_tokens=int(self.tokens.get()))
            text = tok.decode(out_ids[0].tolist())
            self._append(text)
        except Exception as e:
            self._append(f"[error] {e}")
        finally:
            self.run_btn.configure(state=tk.NORMAL)

    def _append(self, text: str) -> None:
        self.out.insert(tk.END, text + "\n")
        self.out.see(tk.END)


def main() -> None:
    gui = PlayGUI()
    gui.root.mainloop()


if __name__ == "__main__":
    main()


