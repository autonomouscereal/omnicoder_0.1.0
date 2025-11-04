import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, help='Path to GGUF model file')
    ap.add_argument('--prompt', type=str, default='Hello, world!')
    ap.add_argument('--n_predict', type=int, default=64)
    ap.add_argument('--n_ctx', type=int, default=4096)
    ap.add_argument('--gpu_layers', type=int, default=0, help='Set >0 if you have GPU acceleration')
    args = ap.parse_args()

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        print('Please install llama-cpp-python: pip install llama-cpp-python')
        return

    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers, logits_all=False)
    out = llm(args.prompt, max_tokens=args.n_predict, temperature=0.7, top_p=0.9)
    print(out['choices'][0]['text'])


if __name__ == "__main__":
    main()
