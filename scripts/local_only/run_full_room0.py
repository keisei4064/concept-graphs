"""Supervise the authorized room0 pilot and 2000-frame local-only experiment.

No Git commands, downloads, environment installation or external inference.
Start after fetch_replica_bulk.py and unit tests pass. Failed stages stop clearly;
relaunch reuses passed stages and complete raw/VLM caches, then recomputes fusion.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback
import urllib.request
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


class Experiment:
    def __init__(self, root, models):
        self.root, self.models = root.resolve(), models.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'stages': {}, 'created': timestamp()}
        self.server = None
        self.child = None
        self.scene = Path('outputs/local-only/assets/data/Replica/room0')
        self.env = dict(os.environ, PYTHONPATH='', PYTHONNOUSERSITE='1')
        self.env.pop('OPENAI_API_KEY', None)
        self.env.pop('ANTHROPIC_API_KEY', None)
        self.write(status='preparing', error=None, supervisor_pid=os.getpid())

    def write(self, **values):
        self.state.update(values, updated=timestamp())
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(self.path)

    def run(self, command, log, experiment=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            self.child = subprocess.Popen(command, env=self.env, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            self.write(child_pid=self.child.pid, command=command, log=str(log))
            last_sample = 0
            while self.child.poll() is None:
                progress = self.scene / 'exps' / experiment / 'progress.json' if experiment else None
                if progress is not None and progress.exists():
                    try:
                        self.write(progress=json.loads(progress.read_text()))
                    except json.JSONDecodeError:
                        pass  # Producer may be replacing its small progress file; next poll retries.
                if time.monotonic() - last_sample >= 60:
                    sample = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                    with (self.root / 'resources.jsonl').open('a') as output:
                        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                        output.write(json.dumps({'time': timestamp(), 'stage': self.state.get('phase'),
                            'gpu_used_free_util': sample.stdout.strip(),
                            'ram_available_kib': int(memory['MemAvailable'].split()[0]),
                            'disk_free_bytes': shutil.disk_usage(self.root).free}) + '\n')
                    last_sample = time.monotonic()
                time.sleep(5)
            code = self.child.returncode
            self.child = None
            if code:
                raise RuntimeError(f'Stage command failed with exit {code}; inspect {log}')

    def audit_server(self, folder):
        self.run(['conda', 'run', '--no-capture-output', '-n', 'conceptgraph-local', 'python',
            'scripts/local_only/audit_network.py', str(folder / 'network.strace')], folder / 'network-audit.json')

    def stop_server(self):
        if self.server is not None:
            server, folder = self.server
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)  # Only the process group created by this supervisor.
                server.wait(timeout=30)
            self.server = None
            if (folder / 'network.strace').exists():
                self.audit_server(folder)
            self.write(ollama_process_group=None)

    def start_server(self, phase, attempt):
        folder = self.root / f'server_{phase}_{attempt}'
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(self.env, CONCEPTGRAPH_OLLAMA_MODELS=str(self.models), CONCEPTGRAPH_OLLAMA_CONTEXT_LENGTH='16384')
        with (folder / 'launcher.log').open('w') as stream:
            process = subprocess.Popen(['bash', 'scripts/local_only/serve_ollama.sh', str(folder)],
                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.server = process, folder
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(30):
            if process.poll() is not None:
                raise RuntimeError(f'Temporary Ollama exited; inspect {folder}')
            try:
                status = json.load(opener.open('http://127.0.0.1:11435/api/status', timeout=2))
                if status.get('cloud', {}).get('disabled') is not True:
                    raise RuntimeError('Temporary Ollama cloud-disable check failed')
                (folder / 'status.json').write_text(json.dumps(status, indent=2))
                self.write(ollama_process_group=process.pid, server_log=str(folder))
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError('Temporary Ollama did not become ready')

    def seed_raw_cache(self):
        source = self.scene / 'exps/local_detections_0_100_1'
        target = self.scene / 'exps/local_room0_full_v1'
        count = 0
        for name in ('detections', 'vis'):
            if not (source / name).exists():
                continue
            for file in (source / name).rglob('*'):
                if not file.is_file() or file.name.endswith('.part') or file.name == '.incomplete':
                    continue
                destination = target / name / file.relative_to(source / name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    os.link(file, destination)
                    count += 1
        self.write(seeded_raw_cache_files=count)

    def stage(self, label, stride, vlm):
        previous = self.state['stages'].get(label, {})
        if previous.get('status') in ('passed', 'passed_with_vlm_issues'):
            if not vlm or previous.get('completion', {}).get('vlm', {}).get('response_policy') == 'upstream':
                print(f'Reusing validated stage: {label}', flush=True)
                return
            print(f'Keeping old {label} artifacts; rerun required for upstream response policy', flush=True)
        attempt = previous.get('attempt', 0) + 1
        folder = self.root / label / f'attempt{attempt}'
        folder.mkdir(parents=True, exist_ok=True)
        exp = f'room0_{label}_attempt{attempt}'
        entry = {'status': 'running', 'attempt': attempt, 'experiment': exp, 'started': timestamp(), 'log_dir': str(folder)}
        self.state['stages'][label] = entry
        self.write(status='running', phase=label, progress=None)
        print(f'Starting {label}: {2000 // stride} frames, experiment {exp}', flush=True)
        if vlm:
            self.start_server(label, attempt)
        reference = label == 'reference_graph'
        config = ('rerun_local_scene_graph_reference' if reference else
                  'rerun_local_room0_full' if vlm else 'rerun_local_replica')
        arguments = [f'--config-name={config}', 'end=2000', f'stride={stride}', f'exp_suffix={exp}',
            'detections_exp_suffix=local_room0_full_v1', 'save_objects_all_frames=false',
            f'periodically_save_pcd={str(not reference).lower()}', 'periodically_save_pcd_interval=250']
        if not vlm:
            arguments += ['use_rerun=false', 'save_rerun=false']
        self.run(['bash', 'scripts/local_only/run_mapping.sh', str(folder)] + arguments,
                 folder / 'launcher.log', experiment=exp)
        self.stop_server()
        output = self.scene / 'exps' / exp
        completion = json.loads((output / 'completion.json').read_text())
        expected = len(range(0, 2000, stride))
        if completion['frames'] != expected or completion['frames_requested'] != expected:
            raise RuntimeError(f'Incomplete experiment: {completion}')
        map_path = output / f'pcd_{exp}.pkl.gz'
        validation = ['conda', 'run', '--no-capture-output', '-n', 'conceptgraph-local', 'python',
            'scripts/local_only/validate_map.py', str(map_path),
            '--output', str(folder / 'map-validation.json')]
        if not vlm:
            validation.append('--no-vlm')
        self.run(validation, folder / 'map-validation.log')
        if vlm:
            self.run(['conda', 'run', '--no-capture-output', '-n', 'conceptgraph-local', 'rerun', 'print',
                str(output / f'rerun_{exp}.rrd')], folder / 'rerun-structure.txt')
        has_issues = completion.get('status') == 'completed_with_vlm_issues'
        entry.update(status='passed_with_vlm_issues' if has_issues else 'passed',
                     finished=timestamp(), completion=completion, map=str(map_path), output=str(output))
        self.write(progress=completion)
        print(f'{"COMPLETED WITH VLM ISSUES" if has_issues else "PASS"} {label}: {completion}', flush=True)

    def cancel(self, signum, frame):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        if self.server is not None and self.server[0].poll() is None:
            os.killpg(self.server[0].pid, signal.SIGTERM)
        self.write(status='interrupted', reason=f'signal {signum}')
        raise SystemExit(128 + signum)

    def execute(self, reference_only=False):
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.cancel)
        try:
            manifest = json.loads(Path('outputs/local-only/assets/replica-room0-manifest.json').read_text())
            if manifest['status'] != 'PASS' or manifest['frames'] != 2000:
                raise RuntimeError('Run CRC-verified 2000-frame acquisition first')
            self.seed_raw_cache()
            stages = ([('reference_graph', 10, True)] if reference_only else
                      [('pilot_detection', 20, False), ('pilot_graph', 20, True),
                       ('full_detection', 1, False), ('full_graph', 1, True)])
            for label, stride, vlm in stages:
                self.stage(label, stride, vlm)
            has_issues = any(stage.get('status') == 'passed_with_vlm_issues'
                             for stage in self.state['stages'].values())
            self.write(status='completed_with_vlm_issues' if has_issues else 'completed',
                       finished=timestamp(), progress=None)
            print('Completed and artifacts validated' + ('; VLM issues recorded' if has_issues else ''), flush=True)
        except Exception as exc:
            self.state['stages'].setdefault(self.state.get('phase', 'setup'), {}).update(status='failed', error=str(exc))
            self.write(status='failed', error=str(exc), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
            raise
        finally:
            self.stop_server()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/local-only/full_room0'))
    parser.add_argument('--ollama-models', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true',
                        help='Run only the official-sampling 200-frame graph; never start the 2000-frame stages')
    args = parser.parse_args()
    Experiment(args.root, args.ollama_models).execute(reference_only=args.reference_only)


if __name__ == '__main__':
    main()
