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
    parser.add_argument('--presentation', action='store_true',
                        help='Show a readable 2D relation diagram beside a minimal 3D node/arrow view')
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
    if args.presentation:
        import numpy as np
        import rerun as rerun_sdk
        import rerun.blueprint as blueprint
        from draw_relation_graph import render
        # Render the same saved map in both views; do not infer new relationships.
        diagram = render(artifact, args.output.parent / (args.output.stem + '_diagram'))
        rr.log('diagram/relations', rr.ImageEncoded(path=str(diagram)))
        rr.log('world', rerun_sdk.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        saved_edges = [edge for edge in artifact['edges']['edges']
                       if edge['num_detections'] >= args.min_edge_detections]
        connected = {edge[key] for edge in saved_edges for key in ('obj1_index', 'obj2_index')}
        centers = np.array([obj['bbox_np'].mean(axis=0) for obj in artifact['objects']])
        for index, obj in enumerate(artifact['objects']):
            points = obj['pcd_np'][::max(1, len(obj['pcd_np']) // 500)]
            rr.log(f'world/context/{index}', rr.Points3D(points, colors=[165, 180, 198, 35], radii=.002))
            if index in connected:
                label = f"#{obj['curr_obj_num']} {obj['class_name']}"
                rr.log(f'world/nodes/{index}', rr.Points3D([centers[index]],
                    radii=.055, colors=[65, 130, 220], labels=[label]))
        for index, edge in enumerate(saved_edges):
            start, end = centers[edge['obj1_index']], centers[edge['obj2_index']]
            rr.log(f'world/relations/{index}', rr.Arrows3D(origins=[start], vectors=[end-start],
                radii=.009, colors=[130, 80, 180] if edge['rel_type'] == 'under' else [35, 125, 210],
                labels=[f"{edge['rel_type']} ({edge['num_detections']})"]))
        rr.send_blueprint(blueprint.Blueprint(blueprint.Horizontal(
            blueprint.Spatial2DView(origin='diagram', name='Relations: object nodes and directed arrows'),
            blueprint.Spatial3DView(origin='world', name='Where the related objects are'),
            column_shares=[3, 2]), collapse_panels=True))
    else:
        orr_log_objs_pcd_and_bbox(objects, classes)
        orr_log_edges(objects, edges, classes, min_detections=args.min_edge_detections)
    rr.disconnect()
    print(f'Saved {len(objects)} objects and {len(edges.edges_by_index)} directed relations: {args.output}')


if __name__ == '__main__':
    main()
