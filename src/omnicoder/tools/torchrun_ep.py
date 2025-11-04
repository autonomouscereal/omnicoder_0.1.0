import os
import argparse
import torch
import torch.distributed as dist


def init_distributed() -> None:
    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)


def all_reduce_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= dist.get_world_size()


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert-parallel launcher for MoE: shards experts across provided devices and runs a target module")
    ap.add_argument('--script', type=str, default='omnicoder.training.pretrain', help='Python module to run (e.g., omnicoder.training.pretrain)')
    ap.add_argument('--script_args', type=str, default='', help='Arguments to pass to the module')
    ap.add_argument('--devices', type=str, default=os.getenv('OMNICODER_EP_DEVICES', 'cuda:0,cuda:1'), help='Comma-separated device list for expert placement')
    ap.add_argument('--router', type=str, default=os.getenv('OMNICODER_ROUTER', ''), help='Optional router override (e.g., llm)')
    ap.add_argument('--init_dist', action='store_true', default=(os.getenv('OMNICODER_INIT_DIST','0')=='1'), help='Initialize torch.distributed. Defaults to off unless running under torchrun (RANK set).')
    args = ap.parse_args()
    # Initialize distributed only when requested or when torchrun provides env
    if args.init_dist or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        init_distributed()
    # Torchrun sets LOCAL_RANK; map CUDA device
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available() and ('LOCAL_RANK' in os.environ or args.init_dist):
        torch.cuda.set_device(local_rank)
    # Configure expert devices (picked up by MoELayer) and optional router override
    if args.devices:
        os.environ['OMNICODER_EXPERT_DEVICES'] = args.devices
    if args.router:
        os.environ['OMNICODER_ROUTER'] = args.router
    # Forward into training module
    import runpy, sys
    sys.argv = [args.script] + [a for a in args.script_args.split(' ') if a]
    runpy.run_module(args.script, run_name='__main__')


if __name__ == '__main__':
    main()


