from pathlib import Path
from unittest.mock import Mock

import numpy as np
import open3d as o3d
import pytest
import supervision as sv

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import filter_gobs, make_detection_list_from_pcd_and_gobs, process_edges
from conceptgraph.utils.general_utils import make_vlm_edges_and_captions
from conceptgraph.utils.general_utils import annotate_for_vlm
from conceptgraph.utils.vlm import empty_captions
from conceptgraph.utils.model_utils import compute_clip_features_batched


@pytest.mark.parametrize("labels", [[], ["chair 0"], ["coffee table 7", "chair 2", "wall 13"]])
def test_disabled_vlm_keeps_every_original_index(labels, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    forbidden = Mock(side_effect=AssertionError("VLM filtering/client must not be touched"))
    monkeypatch.setattr("conceptgraph.utils.general_utils.filter_detections", forbidden)
    result = make_vlm_edges_and_captions(None, None, None, labels, tmp_path, Path("f.jpg"), False, forbidden)
    actual_labels, edges, image, captions = result
    assert actual_labels == labels
    assert edges == [] and image is None
    assert captions == empty_captions(labels)
    assert len(captions) == len(labels)
    forbidden.assert_not_called()


def test_filtered_captions_reach_3d_detections_in_order():
    labels = ["chair 7", "wall 4", "coffee table 13"]
    image = np.zeros((20, 20, 3), np.uint8)
    masks = np.ones((3, 20, 20), dtype=bool)
    gobs = dict(xyxy=np.tile([1, 1, 10, 10], (3, 1)), confidence=np.ones(3),
                class_id=np.array([0, 1, 2]), mask=masks,
                classes=["chair", "wall", "coffee table"], detection_class_labels=labels,
                labels=labels, captions=empty_captions(labels), edges=[],
                image_feats=np.ones((3, 8)), text_feats=np.ones((3, 8)))
    filtered = filter_gobs(gobs, image, skip_bg=True, BG_CLASSES=["wall"],
                           mask_area_threshold=10, max_bbox_area_ratio=1, mask_conf_threshold=.25)
    assert [cap['id'] for cap in filtered['captions']] == ["7", "13"]
    classes = Mock()
    classes.get_classes_arr.return_value = ["chair", "wall", "coffee table"]
    classes.get_bg_classes_arr.return_value = ["wall"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).random((30, 3))))
    spatial = [dict(pcd=pcd, bbox=pcd.get_oriented_bounding_box()) for _ in range(2)]
    detections = make_detection_list_from_pcd_and_gobs(spatial, filtered, Path("frame000000.jpg"), classes, 0)
    assert [det['captions'][0]['id'] for det in detections] == ["7", "13"]
    assert all(det['captions'][0]['caption'] is None for det in detections)
    objects = MapObjectList(detections)
    edges = MapEdgeMapping(objects)
    assert process_edges([0, 1], filtered, 2, objects, edges, 0).edges_by_index == {}


def test_empty_vlm_filter_is_valid(tmp_path):
    classes = Mock(bg_classes=[])
    classes.get_classes_arr.return_value = []
    empty = sv.Detections.empty()
    empty.mask = np.empty((0, 20, 20), dtype=bool)
    labels, edges, image, captions = make_vlm_edges_and_captions(
        np.zeros((20, 20, 3), np.uint8), empty, classes, [], tmp_path, Path("f.jpg"), True, None)
    assert labels == edges == captions == [] and image is None


def test_no_detections_do_not_invoke_clip():
    model = Mock()
    model.visual.output_dim = 8
    crops, features, text = compute_clip_features_batched(
        np.zeros((20, 20, 3), np.uint8), sv.Detections.empty(), model, None, None, [], "cpu")
    assert crops == text == [] and features.shape == (0, 8)
    model.encode_image.assert_not_called()


def test_annotation_keeps_leading_id_inside_image(monkeypatch):
    import cv2
    calls = []
    real_put_text = cv2.putText
    def record(image, text, origin, *args, **kwargs):
        calls.append((text, origin))
        return real_put_text(image, text, origin, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', record)
    mask = np.zeros((100, 300), dtype=bool)
    mask[20:80, :10] = True
    detections = sv.Detections(xyxy=np.array([[0., 20., 10., 80.]]), mask=mask[None],
                               class_id=np.array([0]), confidence=np.array([1.]))
    classes = Mock()
    classes.get_class_color.return_value = [1, 0, 0]
    annotate_for_vlm(np.zeros((100, 300, 3), np.uint8), detections, classes, ['poster 28'])
    assert calls and all(text == '28: poster' and x >= 0 and y >= 0 for text, (x, y) in calls)
