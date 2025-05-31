import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from collections import defaultdict
from cvlabkit.component.base import Metric


def parse_rec(xml_file: str) -> List[Dict[str, Any]]:
    tree = ET.parse(xml_file)
    objects = []
    for obj in tree.findall("object"):
        bbox = obj.find("bndbox")
        objects.append({
            "name": obj.find("name").text,
            "difficult": int(obj.find("difficult").text),
            "bbox": [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ],
        })
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class PascalVOCAPMetric(Metric):
    def __init__(self, cfg):
        root = cfg.root
        split = cfg.split
        self.annotation_dir = os.path.join(root, "Annotations")
        self.image_set_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        if isinstance(cfg.class_names, str): cfg.class_names = [cfg.class_names]
        self.class_names = cfg.class_names
        self.use_07_metric = True

        self.detections = []
        self.gts = {}

    def update(self, **kwargs: Any) -> None:
        """
        kwargs: {
            "image_id": str,
            "boxes": np.ndarray,   # [N, 4]
            "scores": np.ndarray,  # [N]
            "labels": List[str]    # [N]
        }
        """
        self.detections.append(kwargs)

    def compute(self) -> Dict[str, float]:
        with open(self.image_set_file, "r") as f:
            image_ids = [x.strip() for x in f.readlines()]

        for image_id in image_ids:
            xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
            self.gts[image_id] = parse_rec(xml_path)

        ovthresh_list = np.arange(0.5, 1.0, 0.05)
        aps_per_class = defaultdict(list)
        aps_50 = []
        aps_75 = []

        results = {}

        for classname in self.class_names:
            ap_list = []

            for ovthresh in ovthresh_list:
                _, _, ap = self._eval_class(classname, ovthresh=ovthresh)
                ap *= 100
                ap_list.append(ap)

                if np.isclose(ovthresh, 0.5):
                    aps_50.append(ap)
                    results[f"{classname}_AP50"] = ap
                elif np.isclose(ovthresh, 0.75):
                    aps_75.append(ap)
                    results[f"{classname}_AP75"] = ap

            aps_per_class[classname] = ap_list
            results[f"{classname}_AP@[.50:.95]"] = np.mean(ap_list)

        all_ap = [ap for ap_list in aps_per_class.values() for ap in ap_list]

        results["AP@[.50:.95]"] = np.mean(all_ap)
        results["AP50"] = np.mean(aps_50) if aps_50 else 0.0
        results["AP75"] = np.mean(aps_75) if aps_75 else 0.0

        return results

    def _eval_class(self, class_name: str, ovthresh: float = 0.5):
        cls_dets = []
        for det in self.detections:
            for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                if label == class_name:
                    cls_dets.append((det["image_id"], score, box))

        if len(cls_dets) == 0:
            return np.array([]), np.array([]), 0.0

        cls_dets.sort(key=lambda x: -x[1])
        image_ids = [x[0] for x in cls_dets]
        scores = np.array([x[1] for x in cls_dets])
        boxes = np.array([x[2] for x in cls_dets])

        npos = 0
        gt_records = {}
        for image_id in self.gts:
            objs = [obj for obj in self.gts[image_id] if obj["name"] == class_name]
            bbox = np.array([x["bbox"] for x in objs])
            difficult = np.array([x["difficult"] for x in objs]).astype(bool)
            det_flags = [False] * len(objs)
            npos += np.sum(~difficult)
            gt_records[image_id] = {"bbox": bbox, "difficult": difficult, "det": det_flags}

        tp = np.zeros(len(image_ids))
        fp = np.zeros(len(image_ids))

        for i in range(len(image_ids)):
            image_id = str(image_ids[i]).zfill(6)
            bb = boxes[i]
            if image_id not in gt_records:
                fp[i] = 1
                continue

            R = gt_records[image_id]
            BBGT = R["bbox"].astype(float)

            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.)
                ih = np.maximum(iymax - iymin + 1.0, 0.)
                inters = iw * ih
                uni = ((bb[2]-bb[0]+1.) * (bb[3]-bb[1]+1.) +
                       (BBGT[:, 2]-BBGT[:, 0]+1.) * (BBGT[:, 3]-BBGT[:, 1]+1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            else:
                ovmax = 0.

            if ovmax > ovthresh:
                if not R["difficult"][jmax]:
                    if not R["det"][jmax]:
                        tp[i] = 1.
                        R["det"][jmax] = True
                    else:
                        fp[i] = 1.
            else:
                fp[i] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric=self.use_07_metric)

        return rec, prec, ap
