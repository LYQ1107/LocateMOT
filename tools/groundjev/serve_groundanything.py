"""Launch the official custom DLM engine with the project's pinned model bundle."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'external/GroundAnything'
sys.path.insert(0, str(SOURCE))


def main():
    from models.dependency_contract import verify_dependency
    from infer.serve_sglang import build_launch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT/'models/GroundAnything-DLM')
    parser.add_argument('--port', type=int, default=8101)
    parser.add_argument('--mem-fraction-static', type=float, default=.65)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    config = json.loads((ROOT/'configs/groundjev/resources.json').read_text())
    if len(visible) != 1 or not visible[0] or int(visible[0]) not in config['physical_gpu_allowlist']:
        parser.error('DLM TP=1 launch requires exactly one registered physical GPU')
    model = args.model.resolve()
    if not model.is_relative_to(ROOT/'models'):
        parser.error('use the isolated GroundJEV model directory')
    revision = verify_dependency('sglang')
    vendor = SOURCE/'vendor/sglang'
    paths = [str(vendor), str(SOURCE/'infer/engines'), str(SOURCE)]
    sys.path[:0] = paths
    import sglang
    if not Path(sglang.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError('The custom official SGLang source was not loaded')
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    mask = tokenizer.encode('|<MASK>|', add_special_tokens=False)
    with safe_open(model/'model.safetensors', framework='pt') as weights:
        vocab = weights.get_slice('base_model.model.language_model.embed_tokens.weight').get_shape()[0]
    if len(mask) != 1 or mask[0] != vocab-1 or len(tokenizer) != vocab:
        raise RuntimeError('DLM tokenizer and embedding rows differ')
    output = ROOT/'outputs/groundjev/groundanything_service'
    output.mkdir(parents=True, exist_ok=True)
    command, env, settings, algorithm = build_launch(model, output, 'denoise', mask[0], vocab,
        tokenizer.convert_tokens_to_ids('<|im_end|>'), paths, os.environ, port=args.port)
    command[command.index('--mem-fraction-static')+1] = str(args.mem_fraction_static)
    algorithm.write_text(yaml.safe_dump(settings['algorithm_config']))
    receipt = {'status': 'LAUNCHING_NOT_YET_GPU_VALIDATED', 'decoder': 'official_entropy_guided_dlm',
               'checkpoint': str(model), 'physical_gpu': visible, 'command': command, 'revision': revision,
               'memory_fraction_static': args.mem_fraction_static, 'execution': 'bf16_eager_triton',
               'packages': {n: importlib.metadata.version(n) for n in ('torch','transformers','sglang','sgl-kernel')}}
    (output/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    os.chdir(SOURCE)
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
