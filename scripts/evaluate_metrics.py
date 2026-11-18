"""Evaluate a completed best.pt without teacher weights or retraining."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    for name in ['best.pt','config.json']:
        if not (run/name).is_file():
            raise FileNotFoundError(str(run/name))
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8-sig'))
    if cfg['threshold'] != .5:
        raise ValueError('This report requires the saved threshold to be 0.5; do not silently change old protocols')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(ROOT/'run.py'), 'evaluate',
                    '--config', str(run/'config.json'), '--checkpoint', str(run/'best.pt'),
                    '--split', args.split, '--output', str(output/'metrics.json')],
                   cwd=str(ROOT), check=True)
    result = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    rows = result['per_image']
    with (output/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(run_dir=str(run), split=args.split, epoch=result['epoch'],
                   n_images=result['n_images'], threshold=.5, **result['global'])
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print('Reports: '+str(output), flush=True)


if __name__ == '__main__':
    main()
