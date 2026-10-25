"""Render a trusted saved 3D map with directed relation arrows, without a GUI."""
import argparse
import gzip
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map')
    parser.add_argument('output')
    args = parser.parse_args()
    with gzip.open(args.map, 'rb') as stream:
        artifact = pickle.load(stream)
    objects = artifact['objects']
    fig = plt.figure(figsize=(13, 10))
    axis = fig.add_subplot(111, projection='3d')
    centers, all_points = [], []
    for obj in objects:
        points = obj['pcd_np']
        points = points[::max(1, len(points) // 800)]
        all_points.append(points)
        color = obj['inst_color']
        axis.scatter(*points.T, color=color, s=1, alpha=.3)
        center = np.asarray(obj['bbox_np']).mean(axis=0)
        centers.append(center)
        axis.text(*center, f"{obj['curr_obj_num']}: {obj['class_name']}", fontsize=7)
    edges = artifact['edges']['edges']
    for edge in edges:
        start, end = centers[edge['obj1_index']], centers[edge['obj2_index']]
        delta = end - start
        axis.quiver(*start, *delta, color='black', arrow_length_ratio=.15, linewidth=2)
        axis.text(*((start + end) / 2), edge['rel_type'], fontsize=8, color='darkred')
    points = np.concatenate(all_points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, radius = (lo + hi) / 2, max(hi - lo) / 2
    axis.set_xlim(center[0]-radius, center[0]+radius)
    axis.set_ylim(center[1]-radius, center[1]+radius)
    axis.set_zlim(center[2]-radius, center[2]+radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set(xlabel='world x [m]', ylabel='world y [m]', zlabel='world z [m]')
    axis.view_init(elev=25, azim=-65)
    axis.set_title(f"Local Replica scene graph: {len(objects)} objects, {len(edges)} directed edges\n"
                   "Arrows: source to target; structural view, semantic correctness reviewed separately")
    fig.savefig(args.output, dpi=160, bbox_inches='tight')
    print(args.output)


if __name__ == '__main__':
    main()
