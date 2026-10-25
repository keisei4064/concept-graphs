"""Reuse saved M1 detections to create the original numbered VLM annotation."""
import argparse
import json
from pathlib import Path

import cv2
import supervision as sv

from conceptgraph.utils.general_utils import (ObjectClasses, annotate_for_vlm,
                                              filter_detections, load_saved_detections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('detections', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gobs = load_saved_detections(args.detections)
    classes = ObjectClasses('conceptgraph/scannet200_classes.txt', ['wall', 'floor', 'ceiling'], False)
    image = cv2.imread(str(args.image))
    detections = sv.Detections(xyxy=gobs['xyxy'], confidence=gobs['confidence'],
                               class_id=gobs['class_id'], mask=gobs['mask'])
    filtered, labels = filter_detections(image, detections, classes, top_x_detections=150000,
        confidence_threshold=0.00001, given_labels=gobs['detection_class_labels'])
    assert labels, 'No annotated objects remain'
    path = args.output / 'annotated.jpg'
    annotated, _ = annotate_for_vlm(image, filtered, classes, labels, save_path=str(path))
    assert cv2.imwrite(str(path), annotated)
    label_list = [f"{label.rsplit(' ', 1)[1]}: {label.rsplit(' ', 1)[0]}" for label in labels]
    (args.output / 'labels.json').write_text(json.dumps(label_list, indent=2))
    print(f'{len(label_list)} original annotated detections: {path}')


if __name__ == '__main__':
    main()
