"""Export description-key captions from saved responses; no inference or source writes."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re

from conceptgraph.utils.vlm_response_policy import filter_payload


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames, source_hashes = {}, {str(args.map): file_hash(args.map)}
    totals = Counter()
    for path in sorted(args.cache_dir.glob('captions_*.json')):
        record = json.loads(path.read_text())
        if args.experiment not in Path(record.get('image_path', '')).parts:
            continue
        source_hashes[str(path)] = file_hash(path)
        frame = re.search(r'frame\d+', Path(record['image_path']).name).group()
        raw = record['raw_output']
        rows = raw.get('captions') if isinstance(raw, dict) else raw
        normalized, restored_ids = [], set()
        for row in rows:
            normalized_row = dict(row) if isinstance(row, dict) else row
            if (isinstance(row, dict) and 'caption' not in row
                    and isinstance(row.get('description'), str)):
                normalized_row['caption'] = row['description']
                totals['description_items_normalized'] += 1
                restored_ids.add(str(row.get('id')))
            normalized.append(normalized_row)
        accepted, issues = filter_payload('captions', {'captions': normalized}, record['labels'])
        by_id = {row['id']: row for row in accepted['captions']}
        totals['missing_ids_before'] += len(record.get('issues', {}).get('missing_caption_ids', []))
        totals['missing_ids_after'] += len(issues['missing_caption_ids'])
        totals['unparseable_items_retained_in_original_logs'] += sum(
            row.get('reason') == 'unparseable_caption_item'
            for row in record.get('issues', {}).get('rejected_captions', []))
        assert frame not in frames, f'Ambiguous source responses for {frame}'
        frames[frame] = {'frame': frame, 'input_image': record['image_path'],
                        'image_sha256': record['image_sha256'], 'source_record': str(path),
                        'captions': accepted['captions'], 'issues': issues,
                        'description_ids': sorted(restored_ids)}
        # Every recovered value must remain exactly the model's original text.
        for row in normalized:
            if isinstance(row, dict) and str(row.get('id')) in restored_ids and 'description' in row:
                assert row['caption'] == row['description']
    if not frames:
        raise ValueError('No saved caption responses matched the experiment')
    with gzip.open(args.map, 'rb') as stream:
        artifact = pickle.load(stream)  # Trusted local experiment artifact only.
    objects = []
    for obj in artifact['objects']:
        observations = []
        for original, color_path in zip(obj['captions'], obj['color_path']):
            frame = re.search(r'frame\d+', Path(color_path).name).group()
            source = frames.get(frame)
            candidates = {cap['id']: cap for cap in source['captions']} if source else {}
            candidate = candidates.get(original['id'])
            recovered = (not original['caption'] and candidate is not None
                         and bool(candidate['caption'])
                         and original['id'] in source['description_ids'])
            caption = candidate if recovered else original
            totals['map_observations'] += 1
            totals['map_missing_before'] += not bool(original['caption'])
            totals['map_missing_after'] += not bool(caption['caption'])
            totals['map_observations_recovered'] += recovered
            observations.append({'frame': frame, 'detection_id': caption['id'], 'name': caption['name'],
                'caption': caption['caption'], 'recovered_from_description': recovered,
                'source_record': source['source_record'] if source else None})
        objects.append({'map_id': obj['curr_obj_num'], 'uuid': str(obj['id']),
                        'class': obj['class_name'],
                        'original_consolidated_caption': obj.get('consolidated_caption'),
                        'observations': observations})
    assert all(file_hash(Path(path)) == sha for path, sha in source_hashes.items())
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'PASS', 'experiment': args.experiment, 'frames': len(frames),
        'map_objects': len(objects), 'counts': dict(totals),
        'source_map': str(args.map), 'source_sha256': source_hashes,
        'new_inference': False, 'source_files_modified': False,
        'consolidated_captions_regenerated': False, 'original_rrd_modified': False,
        'notes': ['Only description -> caption key normalization; original text and sources preserved.',
                  'Frame detection IDs differ from map object IDs.',
                  'Original consolidated captions are pre-recovery summaries.']}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'frames.json').write_text(json.dumps(list(frames.values()), indent=2))
    (args.output/'objects.json').write_text(json.dumps(objects, indent=2))
    lines = ['# Saved captions recovered without inference', '',
             'Object IDs below match the final map. Detection IDs are local to each frame.',
             'Only field names were normalized. Original consolidated captions have not been regenerated.', '']
    for obj in objects:
        lines += [f"## #{obj['map_id']} {obj['class']}", '']
        if obj['original_consolidated_caption']:
            lines += ['Original summary (before recovery): ' + obj['original_consolidated_caption'], '']
        seen = set()
        for obs in obj['observations']:
            text = obs['caption']
            if not text or text in seen:
                continue
            seen.add(text)
            mark = ' [recovered]' if obs['recovered_from_description'] else ''
            lines.append(f"- {obs['frame']} / detection {obs['detection_id']}{mark}: " + ' '.join(text.split()))
        if not seen:
            lines.append('- No observed caption available.')
        lines.append('')
    (args.output/'captions.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
