import gzip
import importlib.util
import io
import os
import pickle
import zipfile

import numpy as np

from conceptgraph.utils.detection_cache import complete_detection_cache, find_detection_cache
from conceptgraph.utils.general_utils import save_detection_results


def detection():
    return dict(xyxy=np.array([[0, 0, 10, 10]]), confidence=np.array([.9]),
                class_id=np.array([0]), mask=np.ones((1, 10, 10), dtype=bool),
                classes=['chair'], image_feats=np.ones((1, 8)))


def test_partial_cache_does_not_hide_missing_frames(tmp_path):
    folder = tmp_path / 'frame000000'
    save_detection_results(folder, detection())
    assert find_detection_cache(tmp_path, 'frame000000') == folder
    assert find_detection_cache(tmp_path, 'frame000001') is None
    (folder / 'mask.npz').write_bytes(b'interrupted file')
    assert not complete_detection_cache(folder)
    save_detection_results(folder, detection())
    (folder / '.incomplete').write_text('interrupted')
    assert not complete_detection_cache(folder)


def test_atomic_cache_refresh_preserves_hardlinked_source(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    save_detection_results(source, detection())
    target.mkdir()
    for file in source.iterdir():
        os.link(file, target / file.name)
    replacement = detection()
    replacement['class_id'] = np.array([5])
    save_detection_results(target, replacement)
    assert np.load(source / 'class_id.npz')['arr_0'].tolist() == [0]
    assert np.load(target / 'class_id.npz')['arr_0'].tolist() == [5]


def test_legacy_cache_requires_geometry_and_features(tmp_path):
    base = tmp_path / 'frame000000'
    with gzip.open(str(base) + '.pkl.gz', 'wb') as stream:
        pickle.dump(detection(), stream)
    assert complete_detection_cache(base)


def test_batched_zip_ranges_preserve_original_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location('replica_bulk', 'scripts/local_only/fetch_replica_bulk.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payloads = {'Replica/room0/results/frame000000.jpg': b'rgb' * 200,
                'Replica/room0/results/depth000000.png': b'depth' * 200}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('other_scene/file', b'unrelated')
        for name, value in payloads.items():
            archive.writestr(name, value)
        archive.writestr('other_scene/tail', b'tail')
    raw = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        wanted = [archive.getinfo(name) for name in payloads]
        groups = module.group_ranges(wanted, archive.infolist())
        assert len(groups) == 1 and groups[0]['start'] > 0
        group = groups[0]
        fragment = tmp_path / 'range'
        fragment.write_bytes(raw[group['start']:group['end']])
        module.extract_group(archive, group, fragment, tmp_path / 'output')
    assert all((tmp_path / 'output' / name).read_bytes() == value for name, value in payloads.items())


def test_map_save_shares_serialized_nodes_without_modifying_source(tmp_path):
    import uuid
    import open3d as o3d
    import torch
    from unittest.mock import Mock
    from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
    from conceptgraph.utils.general_utils import save_pointcloud
    from omegaconf import OmegaConf
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).random((20, 3))))
    cloud.paint_uniform_color([.5, .5, .5])
    mask = np.ones((20, 20), dtype=bool)
    mapped = MapObjectList([{'id': uuid.uuid4(), 'pcd': cloud, 'bbox': cloud.get_oriented_bounding_box(),
                             'clip_ft': torch.ones(8), 'mask': [mask]}])
    edges = MapEdgeMapping(mapped)
    classes = Mock()
    classes.get_classes_arr.return_value = ['chair']
    classes.get_class_color_dict_by_index.return_value = {'0': [.5, .5, .5]}
    save_pointcloud('test', tmp_path, OmegaConf.create({}), mapped, classes, create_symlink=False, edges=edges)
    with gzip.open(tmp_path / 'pcd_test.pkl.gz', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['objects'] is saved['edges']['objects']
    assert np.array_equal(saved['objects'][0]['mask'][0], mask)
    assert mapped[0]['pcd'] is cloud and mapped[0]['mask'][0] is mask
