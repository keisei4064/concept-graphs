import uuid

import numpy as np
import pytest

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import filter_objects, merge_objects, process_edges


def objects(count):
    return MapObjectList([{'id': uuid.uuid4(), 'curr_obj_num': i, 'class_name': 'test',
                           'num_detections': 1, 'pcd': type('PCD', (), {'points': [0]})()}
                          for i in range(count)])


def test_relation_uses_surviving_3d_detection_ids_and_premerge_object_count(caplog):
    mapped = objects(2)  # Existing object 0, newly appended object 1.
    edges = MapEdgeMapping(mapped)
    gobs = {'detection_class_labels': ['chair 5', 'discarded 7', 'table 13'],
            'edges': [('5', 'on top of', '13'), ('7', 'under', '13')]}
    with caplog.at_level('INFO'):
        process_edges([0, None], gobs, 1, mapped, edges, 1, detection_labels=['5', '13'])
    assert set(edges.edges_by_index) == {(0, 1)}
    assert edges.edges_by_index[(0, 1)].rel_type == 'on top of'
    assert 'source removed by geometry/filter' in caplog.text


def test_filter_reindexes_internal_fields_and_uuids_then_clears_all():
    mapped = objects(3)
    mapped[0]['num_detections'] = 0
    edges = MapEdgeMapping(mapped)
    edges.add_or_update_edge(1, 2, 'under', 0)
    filtered = filter_objects(0, 1, mapped, edges)
    edge = edges.edges_by_index[(0, 1)]
    assert (edge.obj1_idx, edge.obj2_idx) == (0, 1)
    assert (filtered[0]['id'], filtered[1]['id']) in edges.edges_by_uuid
    assert filter_objects(0, 2, filtered, edges) == []
    assert edges.objects == [] and edges.edges_by_index == edges.edges_by_uuid == {}


def test_merge_reindexes_uuids_against_the_new_object_list(monkeypatch):
    mapped = objects(3)
    edges = MapEdgeMapping(mapped)
    edges.add_or_update_edge(1, 2, 'on top of', 0)
    kept = MapObjectList(mapped[1:])
    monkeypatch.setattr('conceptgraph.slam.utils.compute_overlap_matrix_general', lambda **kwargs: np.zeros((3, 3)))
    monkeypatch.setattr('conceptgraph.slam.utils.merge_overlap_objects', lambda **kwargs: (kept, [None, 0, 1]))
    result, edges = merge_objects(.7, .7, .7, mapped, .025, False, .1, 10, 'overlap', 'cpu', True, edges)
    assert result is kept
    assert set(edges.edges_by_uuid) == {(kept[0]['id'], kept[1]['id'])}
    edge = edges.edges_by_index[(0, 1)]
    assert (edge.obj1_idx, edge.obj2_idx) == (0, 1)


@pytest.mark.parametrize('with_edges', [False, True])
def test_empty_merge_preserves_return_contract(with_edges):
    mapped = MapObjectList()
    edges = MapEdgeMapping(mapped)
    result = merge_objects(.7, .7, .7, mapped, .025, False, .1, 10, 'overlap', 'cpu', with_edges, edges)
    assert result == ((mapped, edges) if with_edges else mapped)
