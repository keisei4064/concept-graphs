"""Reload a trusted local map and reject empty/nonfinite/dangling artifacts."""
import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path)
    parser.add_argument("--require-edges", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with gzip.open(args.map, "rb") as stream:
        artifact = pickle.load(stream)
    serialized = artifact['objects']
    assert serialized, "Map is empty"
    objects = MapObjectList()
    objects.load_serializable(serialized)
    assert len({str(obj['id']) for obj in objects}) == len(objects), "Duplicate UUIDs"
    assert len({obj['curr_obj_num'] for obj in objects}) == len(objects), "Duplicate display IDs"
    report = {"objects": len(objects), "detections": 0, "frames_with_associations": set(), "nodes": [],
              "caption_name_differences": 0, "missing_captions": 0}
    for obj in objects:
        points = np.asarray(obj['pcd'].points)
        bbox = np.asarray(obj['bbox'].get_box_points())
        features = obj['clip_ft'].cpu().numpy()
        assert points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
        assert np.isfinite(points).all() and np.abs(points).max() < 100
        assert bbox.shape == (8, 3) and np.isfinite(bbox).all()
        assert features.ndim == 1 and np.isfinite(features).all() and np.linalg.norm(features) > 0
        assert np.isclose(np.linalg.norm(features), 1, atol=1e-4)
        assert len(obj['captions']) == len(obj['image_idx']) == obj['num_detections']
        assert all(set(cap) == {"id", "name", "caption"} for cap in obj['captions'])
        assert all(isinstance(cap['id'], str) and isinstance(cap['name'], str)
                   and (cap['caption'] is None or isinstance(cap['caption'], str)) for cap in obj['captions'])
        report['caption_name_differences'] += sum(
            cap['name'] != artifact['class_names'][class_id]
            for cap, class_id in zip(obj['captions'], obj['class_id']))
        report['missing_captions'] += sum(cap['caption'] is None for cap in obj['captions'])
        if args.no_vlm:
            assert all(cap['caption'] is None for cap in obj['captions'])
            assert obj['consolidated_caption'] is None
        report['detections'] += obj['num_detections']
        report['frames_with_associations'].update(obj['image_idx'])
        report['nodes'].append({"id": str(obj['id']), "display_id": obj['curr_obj_num'],
                                "class": obj['class_name'], "points": len(points),
                                "feature_dim": len(features), "center": obj['bbox'].center.tolist(),
                                "detections": obj['num_detections']})
    edge_data = artifact.get('edges') or {"edges": [], "objects": []}
    edges = edge_data['edges']
    if edge_data['objects']:
        assert [obj['id'] for obj in edge_data['objects']] == [obj['id'] for obj in objects]
        reloaded = MapEdgeMapping(objects)
        reloaded.load_serializable(edge_data, load_objects=False)
        assert len(reloaded.edges_by_index) == len(edges)
        for (a, b), edge in reloaded.edges_by_index.items():
            assert (a, b) == (edge.obj1_idx, edge.obj2_idx)
            assert (objects[a]['id'], objects[b]['id']) in reloaded.edges_by_uuid
            assert np.isfinite(reloaded.get_edge_endpoints(a, b)).all()
    for edge in edges:
        a, b = edge['obj1_index'], edge['obj2_index']
        assert isinstance(a, int) and isinstance(b, int) and a != b
        assert 0 <= a < len(objects) and 0 <= b < len(objects)
        assert edge['rel_type'] in {"on top of", "under"}
        assert edge['num_detections'] > 0
    report['edges'] = edges
    report['frames_with_associations'] = sorted(report['frames_with_associations'])
    report['status'] = 'PASS'
    if args.require_edges:
        assert edges, "No serialized edges"
    if args.no_vlm:
        assert not edges
    output = args.output or args.map.with_suffix('.validation.json')
    output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({key: report[key] for key in ('status', 'objects', 'detections', 'frames_with_associations')}))


if __name__ == "__main__":
    main()
