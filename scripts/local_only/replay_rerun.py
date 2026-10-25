"""Replay a trusted saved map/graph through the existing Rerun logging helpers."""
import argparse
import gzip
from pathlib import Path
import pickle
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--min-edge-detections', type=int, default=1)
    args = parser.parse_args()
    from conceptgraph.utils.local_runtime import install_local_network_guard
    install_local_network_guard(args.output.with_suffix('.network.jsonl'))
    from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
    from conceptgraph.utils.optional_rerun_wrapper import OptionalReRun, orr_log_objs_pcd_and_bbox, orr_log_edges
    with gzip.open(args.map, 'rb') as stream:
        artifact = pickle.load(stream)
    edges = MapEdgeMapping(MapObjectList())
    edges.load_serializable(artifact['edges'])
    objects = edges.objects
    def color(name):
        return artifact['class_colors'][str(artifact['class_names'].index(name))]
    classes = SimpleNamespace(get_class_color=color)
    rr = OptionalReRun()
    rr.set_use_rerun(True)
    rr.init('local_saved_scene_graph')
    rr.save(str(args.output))
    rr.set_time_sequence('frame', max(index for obj in objects for index in obj['image_idx']))
    orr_log_objs_pcd_and_bbox(objects, classes)
    orr_log_edges(objects, edges, classes, min_detections=args.min_edge_detections)
    rr.disconnect()
    print(f'Saved {len(objects)} objects and {len(edges.edges_by_index)} directed relations: {args.output}')


if __name__ == '__main__':
    main()
