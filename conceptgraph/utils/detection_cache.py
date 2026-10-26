"""Recognize complete raw detection caches independently of optional VLM fields."""
import gzip
from pathlib import Path
import pickle
import zipfile

REQUIRED_FIELDS = {'xyxy', 'confidence', 'class_id', 'mask', 'classes', 'image_feats'}


def complete_detection_cache(base):
    base = Path(base)
    if base.is_dir():
        if (base / '.incomplete').exists():
            return False
        for key in REQUIRED_FIELDS:
            array = base / f'{key}.npz'
            other = base / f'{key}.pkl.gz'
            if array.is_file():
                if not zipfile.is_zipfile(array):
                    return False
            elif other.is_file():
                try:
                    with gzip.open(other, 'rb') as stream:
                        pickle.load(stream)
                except (OSError, EOFError, pickle.UnpicklingError):
                    return False
            else:
                return False
        return True
    legacy = Path(str(base) + '.pkl.gz')
    if legacy.is_file():
        try:
            with gzip.open(legacy, 'rb') as stream:
                result = pickle.load(stream)
            return isinstance(result, dict) and REQUIRED_FIELDS.issubset(result)
        except (OSError, EOFError, pickle.UnpicklingError):
            return False
    return False


def find_detection_cache(directory, stem):
    candidates = [Path(directory) / stem]
    if stem.isdecimal():
        candidates.append(Path(directory) / f'{int(stem):06d}')
    return next((base for base in candidates if complete_detection_cache(base)), None)
