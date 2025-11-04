import argparse

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception:
    quantize_dynamic = None
    QuantType = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    ap.add_argument('--out', type=str, required=True, help='Output path for quantized ONNX')
    ap.add_argument('--dtype', type=str, default='int8', choices=['int8'], help='Dynamic quantization data type')
    args = ap.parse_args()

    if quantize_dynamic is None:
        print('onnxruntime and onnxruntime-tools are required: pip install onnxruntime onnxruntime-tools')
        return

    qtype = QuantType.QInt8
    quantize_dynamic(model_input=args.model, model_output=args.out, weight_type=qtype)
    print(f'Saved quantized ONNX to {args.out}')


if __name__ == '__main__':
    main()


