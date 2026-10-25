#!/usr/bin/env python3
"""Read-only local ACT training dashboard. Standard library only; no GPU imports."""
import argparse
import json
import math
from pathlib import Path
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[2]
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)'
METRIC = re.compile(r'([A-Za-z][A-Za-z0-9_/]*):\s*(' + NUMBER + r')([KM]?)', re.I)
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def snapshot(run):
    candidates = [run / 'train.log', Path(str(run) + '.log')]
    existing = [p for p in candidates if p.is_file()]
    log = max(existing, key=lambda p: p.stat().st_mtime) if existing else None
    result = {'run': str(run), 'log': str(log) if log else None,
              'points': [], 'validation': [], 'checkpoints': [], 'warnings': [],
              'target': None, 'age_s': None, 'finished': False, 'split': None}
    if log:
        text = ANSI.sub('', log.read_text(errors='replace')).replace('\r', '\n')
        points, validation = {}, {}
        frequencies = re.findall(r'[\"\']log_freq[\"\']:\s*(\d+)', text)
        frequency = int(frequencies[-1]) if frequencies else 20
        progress_step = None
        for line in text.splitlines():
            progress = re.search(r'Training:.*?\|\s*(\d+)/(\d+)', line)
            if progress:
                progress_step = int(progress[1])
            if re.search(r'\bstep:\s*\d', line) and 'loss:' in line:
                row = {}
                for key, value, suffix in METRIC.findall(line):
                    number = float(value) * {'': 1, 'K': 1000, 'M': 1000000}[suffix.upper()]
                    if not math.isfinite(number):
                        result['warnings'].append(f'Non-finite {key} at training step {row.get("step", "?")}')
                        number = None
                    row[key] = number
                # Native MetricsTracker rounds 1020, 1040, ... to "1K".
                # tqdm keeps exact counts and may lag the log by an update;
                # metric logs occur at multiples of the configured frequency.
                if (re.search(r'\bstep:\s*[\d.]+[KM]', line, re.I)
                        and progress_step is not None and frequency > 0):
                    row['step'] = ((progress_step + frequency - 1) // frequency) * frequency
                if row.get('step') is not None:
                    points[int(row['step'])] = row
            match = re.search(r'step\s+(\d+):\s*eval_loss=(' + NUMBER + ')', line, re.I)
            if match:
                value = float(match[2])
                if math.isfinite(value):
                    validation[int(match[1])] = {'step': int(match[1]), 'loss': value}
                else:
                    result['warnings'].append(f'Non-finite validation loss at step {match[1]}')
        result['points'] = [points[k] for k in sorted(points)]
        result['validation'] = [validation[k] for k in sorted(validation)]
        targets = re.findall(r'[\"\']steps[\"\']:\s*(\d+)', text)
        if targets:
            result['target'] = int(targets[-1])
        result['age_s'] = max(0, time.time() - log.stat().st_mtime)
        result['finished'] = text.rfind('End of training') > max(
            text.rfind('eval_loss='), text.rfind('Training:'), text.rfind(' loss:'))
    checkpoint_root = run / 'checkpoints'
    if checkpoint_root.is_dir():
        result['checkpoints'] = sorted(int(p.name) for p in checkpoint_root.iterdir()
                                       if p.name.isdigit() and (p / 'pretrained_model/model.safetensors').is_file()
                                       and (p / 'training_state/training_step.json').is_file()
                                       and (p / 'training_state/optimizer_state.safetensors').is_file())
    report = run / 'split_and_normalization.json'
    if report.is_file():
        try:
            data = json.loads(report.read_text())
            result['split'] = {k: data[k] for k in ('train_frames', 'validation_frames', 'train_episodes', 'validation_episodes')}
        except (ValueError, KeyError):
            pass  # The trainer may be writing the first report right now.
    vals = result['validation']
    if len(vals) >= 4:
        best = min(v['loss'] for v in vals)
        if all(v['loss'] > best for v in vals[-3:]):
            result['warnings'].append('The last three validation evaluations are above the run minimum. Compare the trend; this alone does not prove overfitting.')
    result['warnings'] = list(dict.fromkeys(result['warnings']))
    return result


def serve(args):
    run = args.run_dir.resolve()
    html = Path(__file__).with_suffix('.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                body, kind = html, 'text/html; charset=utf-8'
            elif self.path == '/api/metrics':
                try:
                    body = json.dumps(snapshot(run), allow_nan=False).encode()
                except OSError as error:
                    body = json.dumps({'error': str(error)}).encode()
                kind = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Training dashboard: http://{args.host}:{args.port}\nWatching: {run}\nRead-only; refreshes every 5 seconds.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=REPO / 'outputs/train/act-cheese-remote-crop15-v1')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args)
