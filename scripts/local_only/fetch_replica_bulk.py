"""ONLINE setup: fetch missing room0 files in batched HTTP ranges with ZIP CRC checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zlib

import requests
from remotezip import RemoteZip

URL = 'https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip'


def group_ranges(infos, all_infos, max_bytes=64 * 1024**2, max_gap=1024**2):
    ordered = sorted(all_infos, key=lambda item: item.header_offset)
    next_offset = {a.header_offset: b.header_offset for a, b in zip(ordered, ordered[1:])}
    groups = []
    for info in sorted(infos, key=lambda item: item.header_offset):
        # _end_offset comes from the original central directory; includes descriptors.
        end = next_offset.get(info.header_offset, info._end_offset)
        if end is None:
            raise RuntimeError(f'Cannot establish safe range end for {info.filename}')
        if (groups and info.header_offset - groups[-1]['end'] <= max_gap
                and end - groups[-1]['start'] <= max_bytes):
            groups[-1]['end'] = end
            groups[-1]['members'].append(info)
        else:
            groups.append({'start': info.header_offset, 'end': end, 'members': [info]})
    return groups


def fetch_range(group, cache):
    start, end = group['start'], group['end']
    path = cache / f'{start}-{end}.ziprange'
    expected = end - start
    if path.exists() and path.stat().st_size == expected:
        return path
    part = path.with_suffix('.part')
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(3):
            offset = part.stat().st_size if part.exists() else 0
            if offset > expected:
                raise RuntimeError(f'Oversized partial range: {part}')
            if offset == expected:
                part.replace(path)
                return path
            try:
                headers = {'Range': f'bytes={start + offset}-{end - 1}', 'Accept-Encoding': 'identity'}
                with session.get(URL, headers=headers, stream=True, timeout=(20, 180)) as response:
                    response.raise_for_status()
                    actual = response.headers.get('Content-Range', '')
                    if response.status_code != 206 or not actual.startswith(f'bytes {start + offset}-{end - 1}/'):
                        raise RuntimeError(f'Server ignored/changed requested byte range: {actual}')
                    with part.open('ab') as stream:
                        for chunk in response.iter_content(1024**2):
                            stream.write(chunk)
                if part.stat().st_size != expected:
                    raise RuntimeError(f'Incomplete HTTP range: {part}')
                part.replace(path)
                return path
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    raise RuntimeError(f'Range download failed: {path}')


def extract_group(archive, group, path, destination):
    remote_fp = archive.fp
    try:
        with path.open('rb') as local:
            archive.fp = local
            for original in group['members']:
                info = copy.copy(original)
                info.header_offset -= group['start']
                if info._end_offset is not None:
                    info._end_offset -= group['start']
                target = destination / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source:
                    payload = source.read()  # zipfile validates decompression, length and CRC.
                if target.exists():
                    if target.stat().st_size != info.file_size or zlib.crc32(target.read_bytes()) != info.CRC:
                        raise RuntimeError(f'Existing input differs; refusing to overwrite: {target}')
                    continue
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                try:
                    os.link(temporary, target)  # Atomic create; never replace existing input.
                finally:
                    temporary.unlink()
    finally:
        archive.fp = remote_fp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/local-only/assets'))
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.frames < 1 or not 1 <= args.workers <= 8:
        parser.error('frames must be positive and workers must be between 1 and 8')
    root = args.root.resolve()
    cache = root / 'replica-range-cache'
    cache.mkdir(parents=True, exist_ok=True)
    names = ['Replica/room0/traj.txt'] + [
        f'Replica/room0/results/{kind}{index:06d}.{ext}' for index in range(args.frames)
        for kind, ext in [('frame', 'jpg'), ('depth', 'png')]]
    with RemoteZip(URL, timeout=90) as archive:
        infos = [archive.getinfo(name) for name in names]
        missing = []
        for info in infos:
            path = root / 'data' / info.filename
            if path.exists():
                if path.stat().st_size != info.file_size or zlib.crc32(path.read_bytes()) != info.CRC:
                    raise RuntimeError(f'Existing input CRC differs: {path}')
            else:
                missing.append(info)
        groups = group_ranges(missing, archive.infolist())
        print(json.dumps({'missing_files': len(missing), 'range_requests': len(groups),
            'download_MiB': round(sum(g['end'] - g['start'] for g in groups) / 1024**2, 1),
            'workers': args.workers}), flush=True)
        downloaded = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fetch_range, group, cache): index for index, group in enumerate(groups)}
            for future in as_completed(pending):
                index = pending[future]
                downloaded[index] = future.result()
                print(f'Fetched range {len(downloaded)}/{len(groups)}', flush=True)
        for index, group in enumerate(groups):
            extract_group(archive, group, downloaded[index], root / 'data')
            print(f'CRC-verified extraction {index + 1}/{len(groups)}', flush=True)
        manifest = {'source': URL, 'frames': args.frames, 'status': 'PASS', 'files': {}}
        for info in infos:
            path = root / 'data' / info.filename
            assert path.stat().st_size == info.file_size
            manifest['files'][info.filename] = {'crc32': f'{info.CRC:08x}',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': info.file_size}
        (root / 'replica-room0-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'PASS: {args.frames} real RGB-D pairs, original trajectory and ZIP CRC verified', flush=True)


if __name__ == '__main__':
    main()
