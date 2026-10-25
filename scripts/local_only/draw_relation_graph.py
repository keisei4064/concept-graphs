"""Draw the saved directed scene graph as readable 2D Graphviz SVG/PNG.

Only saved relations are drawn. By default, isolated objects are listed separately
so they do not overwhelm the actual graph. No inference or mapping is rerun.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import subprocess


def build_dot(artifact, include_isolated=False):
    objects = artifact['objects']
    edges = artifact['edges']['edges']
    connected = {edge[key] for edge in edges for key in ('obj1_index', 'obj2_index')}
    assert all(0 <= index < len(objects) for index in connected)
    isolated = [index for index in range(len(objects)) if index not in connected]
    shown = list(range(len(objects))) if include_isolated else sorted(connected)
    if not shown:
        shown = list(range(len(objects)))
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    model = artifact['cfg'].get('vlm', {}).get('model', 'saved map')
    title = (f"SCENE GRAPH  /  {model}  /  {artifact['cfg'].get('scene_id', 'scene')}\n"
             f"{len(objects)} objects  |  {len(edges)} saved relations  |  "
             f"{len(connected)} connected objects\n"
             "Read arrows: SOURCE -- relation --> TARGET.  Labels include observation counts.")
    lines = ['digraph scene_graph {',
        f'graph [rankdir=LR, bgcolor="#f7f9fc", pad="0.4", nodesep="0.7", ranksep="1.3", '
        f'fontname="DejaVu Sans", fontsize=17, fontcolor="#18243b", labelloc=t, label={quote(title)}];',
        'node [shape=box, style="rounded,filled", fillcolor="#e1ebff", color="#5680bf", '
        'penwidth=1.6, fontname="DejaVu Sans", fontsize=16, margin="0.22,0.16"];',
        'edge [fontname="DejaVu Sans", fontsize=13, penwidth=2.4, arrowsize=1.0, '
        'color="#2c72b8", fontcolor="#254a72"];']
    for index in shown:
        obj = objects[index]
        label = f"{obj['class_name']}\n#{obj['curr_obj_num']}  /  {obj['num_detections']} observations"
        tooltip = obj.get('consolidated_caption') or 'No caption'
        extra = ', fillcolor="#eef0f4", color="#aeb8c8"' if index in isolated else ''
        lines.append(f'n{index} [label={quote(label)}, tooltip={quote(tooltip)}{extra}];')
    for edge in edges:
        source, target = edge['obj1_index'], edge['obj2_index']
        assert source != target and edge['rel_type'] in {'on top of', 'under'}
        count = edge['num_detections']
        label = f"{edge['rel_type']}\n({count} observation{'s' if count != 1 else ''})"
        color = '#8056a6' if edge['rel_type'] == 'under' else '#2c72b8'
        lines.append(f'n{source} -> n{target} [label={quote(label)}, color="{color}", fontcolor="{color}"];')
    lines.append('}')
    return '\n'.join(lines) + '\n', connected, isolated


def render(artifact, output_dir, include_isolated=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dot, connected, isolated = build_dot(artifact, include_isolated)
    source = output_dir / 'relations.dot'
    source.write_text(dot)
    for extension in ('svg', 'png'):
        subprocess.run(['dot', f'-T{extension}', str(source), '-o', str(output_dir / f'relations.{extension}')],
                       check=True, capture_output=True, text=True)
    objects = artifact['objects']
    inventory = [{'map_index': index, 'id': str(obj['id']), 'display_id': obj['curr_obj_num'],
                  'name': obj['class_name'], 'caption': obj.get('consolidated_caption'),
                  'has_saved_relation': index in connected} for index, obj in enumerate(objects)]
    (output_dir / 'objects.json').write_text(json.dumps(inventory, indent=2))
    (output_dir / 'isolated_objects.txt').write_text(
        f'{len(isolated)} objects have no saved relation; omitted from the focused diagram.\n' +
        '\n'.join(f"#{objects[i]['curr_obj_num']} {objects[i]['class_name']}" for i in isolated) + '\n')
    return output_dir / 'relations.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--include-isolated', action='store_true')
    args = parser.parse_args()
    with gzip.open(args.map, 'rb') as stream:
        artifact = pickle.load(stream)
    path = render(artifact, args.output_dir, args.include_isolated)
    print(path)


if __name__ == '__main__':
    main()
