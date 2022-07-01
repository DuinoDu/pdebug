"""Temporal code."""
import sys
import tempfile

# from pycocotools.coco import COCO
# from mmdet.datasets import CocoDataset
# from mmdet.datasets.pipelines import LoadImageFromFile
CocoDataset = object


__all__ = ['COCODataset']


class COCODataset(CocoDataset):

    def __init__(self, ann_file, **kwargs):
        if 'classes' not in kwargs and kwargs.pop('custom_coco_classes', False):
            kwargs['classes'] = self.get_classes_from_anno(ann_file)
        pipeline = kwargs.pop('pipeline') if 'pipeline' in kwargs \
                else [LoadImageFromFile()]
        super(COCODataset, self).__init__(ann_file, pipeline, **kwargs)

    def get_classes_from_anno(self, ann_file):
        """
        get classes from anno, it may be slow.
        """
        _stdout = sys.stdout
        with tempfile.TemporaryFile('wt') as sys.stdout:
            coco = COCO(ann_file)
        sys.stdout = _stdout
        classes = [coco.cats[_id]['name'] for _id in coco.cats]
        print(f'classes: {classes}')
        return classes
