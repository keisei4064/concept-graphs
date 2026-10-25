"""Real Ollama three-task test. Never downloads or changes a service/model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from conceptgraph.utils.local_runtime import install_local_network_guard
    install_local_network_guard(args.output / 'network.jsonl')
    from conceptgraph.utils.ollama_vlm import LocalOllamaClient
    labels = json.loads((args.input / 'labels.json').read_text())
    started = time.monotonic()
    client = LocalOllamaClient(base_url=args.base_url, model=args.model,
                                log_dir=args.output / 'responses', reuse_cache=False)
    try:
        image = args.input / 'annotated.jpg'
        relations = client.relations(image, labels)
        captions = client.captions(image, labels)
        chosen = next(cap for cap in captions if cap['caption'])
        consolidated = client.consolidate([chosen])
        ps = client._get('/api/ps')
        (args.output / 'ps.json').write_text(json.dumps(ps, indent=2))
        env = dict(os.environ, OLLAMA_HOST=client.origin)
        cli_ps = subprocess.run(['ollama', 'ps'], env=env, capture_output=True, text=True, check=True).stdout
        (args.output / 'ollama-ps.txt').write_text(cli_ps)
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
        (args.output / 'nvidia-smi.txt').write_text(gpu)
        summary = {'model': args.model, 'digest': client.digest, 'base_url': client.base_url,
            'elapsed_seconds': time.monotonic() - started, 'relations': relations,
            'captions': captions, 'consolidation_input_id': chosen['id'],
            'consolidated_caption': consolidated, 'gpu_fully_loaded': '100% GPU' in cli_ps}
        summary['status'] = 'PASS' if relations and captions and consolidated and summary['gpu_fully_loaded'] else 'FAIL'
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k not in ('captions', 'relations')}, indent=2))
        if summary['status'] != 'PASS':
            raise RuntimeError('One or more three-task/GPU acceptance checks failed; inspect summary.json')
    finally:
        client.close()


if __name__ == '__main__':
    main()
