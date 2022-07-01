"""
COCOWriter and save_to functions.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime

from .coco import _isArrayLike
from .utils import roidb_to_cocoDt

__all__ = ['COCOWriter', 'save_to_cocoDt', 'save_to_cocoGt']


logger = logging.getLogger(__name__)


class COCOWriter:

    """Convert roidb to coco(gt format).

    Parameters
    ----------
    actions : bool
        add actions annotations
    videos : bool
        add videos annotations
    person: bool
        set categories with person only
    image_ext: str
        default image ext used in images['file_name'] when there is no ext

    Example
    -------
    >>> from piata.coco import COCOWriter
    >>> writer = COCOWriter(videos=True, person=True)
    >>> writer.add_roidb(roidb)
    >>> writer.save('annots.json')

    """

    def __init__(
        self,
        *,
        actions=False,
        videos=False,
        person=False,
        dummy_object=False,
        image_ext='.jpg',
        image_height=None,
        image_width=None,
    ):
        if actions: self._actions = list()
        if videos: self._videos = list()
        self._image_ext = image_ext
        self._image_height = image_height
        self._image_width = image_width
        self._person = person
        self._dummy_object = dummy_object
        self._categories = list()
        self._info = dict()
        self._images = defaultdict(dict)       # id: image
        self._annotations = defaultdict(dict)  # id: anno
        # initialize info
        self._info[u'description'] = 'Created by piata.'
        self._info[u'version'] = 1.0
        self._info[u'year'] = int(datetime.today().isoformat()[:4])
        self._info[u'contributor'] = os.environ['USER']
        self._info[u'date_created'] = datetime.today().isoformat()[:10].replace('-','/')
        # initialize categories
        if self._person:
            self._categories = [{'id': 1, 'name': 'person'}]
        if self._dummy_object:
            self._categories = [{'id': 1, 'name': 'dummy_object'}]
        self._anno_id = 1

    def _is_coco_image_name(self, image_name):
        name = os.path.splitext(image_name)[0]
        # if len(name) != 12:
        #     return False, None
        try:
            _id = int(name)
        except Exception as e:
            return False, None
        return True, _id

    def _get_image_id(self, image_name, prefix=None):
        """
        Get image_id given image_name, starts from 1.
        For video-type data, please set prefix.

        """
        if not hasattr(self, '_image_name_to_id'):
            self._image_name_to_id = dict()
        image_key = prefix + '_' + image_name if prefix else image_name
        if image_key not in self._image_name_to_id:
            is_coco_name, image_id = self._is_coco_image_name(image_name)
            if not is_coco_name:
                image_id = len(self._image_name_to_id) + 1
            self._image_name_to_id[image_key] = image_id
        return self._image_name_to_id[image_key]

    def _get_video_id(self, videoname):
        """
        Get video_id given videoname, starts from 1.
        """
        if not hasattr(self, '_video_name_to_id'):
            self._video_name_to_id = dict()
        if videoname not in self._video_name_to_id:
            video_id = len(self._video_name_to_id) + 1
            self._video_name_to_id[videoname] = video_id
        return self._video_name_to_id[videoname]

    def _set_video_in_image(self, image, videoname):
        """
        Set video tag in image.

        "video_id": 1           # 属于哪个video
        "frame_id": 1,          # 从video获取frame
        "prev_image_id": 300,   # 前一帧的image_id
        "next_image_id": 302,   # 后一帧的image_id

        """
        video_id = self._get_video_id(videoname)
        image_name = image['file_name']
        try:
            if '_' in image['file_name']:
                frame_id = int(os.path.splitext(image['file_name'])[0].split('_')[-1])
            else:
                frame_id = int(os.path.splitext(image['file_name'])[0])
        except Exception as e:
            logging.error('invalid image_name when using videos')
            raise e
        prev_image_id = self._get_image_id(image_name, prefix=videoname) - 1
        next_image_id = self._get_image_id(image_name, prefix=videoname) + 1
        image.update({
            "video_id": video_id,
            "frame_id": frame_id,
            "prev_image_id": prev_image_id,
            "next_image_id": next_image_id,
        })

    def add_roidb(self, roidb, videoname=None):
        """
        Add roidb in coco.

        Parameters
        ----------
        roidb : list
            roi list
        videoname: str
            videoname, only used when using videos

        """
        logger.info('adding roidb ...')
        assert isinstance(roidb, list)
        # avoid loop-import
        from piata import Input
        roidb_roiwise = Input.imgwise_to_roiwise(roidb)
        for roi in roidb_roiwise:
            image_name = roi['image_name']
            if os.path.splitext(image_name)[1] == '':
                image_name += self._image_ext
            # update self._images
            image_id = self._get_image_id(image_name, prefix=videoname)
            if image_id not in self._images:
                image = {
                    'id': image_id,
                    'file_name': image_name,
                }
                if self._image_height:
                    image.update({'height': self._image_height})
                if self._image_width:
                    image.update({'width': self._image_width})
                if 'image_height' in roi:
                    image.update({'height': roi['image_height']})
                if 'image_width' in roi:
                    image.update({'width': roi['image_width']})
                if hasattr(self, '_videos'):
                    self._set_video_in_image(image, videoname)
                self._images[image_id] = image
            # update self._annotations
            anno = {
                'id': self._anno_id,
                'image_id': image_id,
            }
            for key in roi:
                if key == 'image_name': continue
                if key == 'image_height': continue
                if key == 'image_width': continue
                if key == 'unknown': continue
                if key == 'boxes':
                    assert roi[key].ndim == 1
                    x1, y1, x2, y2 = roi[key][:4]
                    bbox = [x1, y1, x2-x1, y2-y1]
                    bbox = [float(x) for x in bbox]
                    anno.update({'bbox': bbox})
                    continue
                if key in ('segmentation', 'masks', 'gt_masks'):
                    anno.update({'segmentation': roi[key]})
                    continue
                if _isArrayLike(roi[key]):
                    value = [float(x) for x in roi[key]]
                else:
                    try:
                        value = float(roi[key])
                    except ValueError as e:
                        value = roi[key]
                anno.update({key: value})
            if 'category_id' not in anno:
                if self._person:
                    anno.update({'category_id': 1})
                elif self._dummy_object:
                    anno.update({'category_id': 1})
                else:
                    # get category info from classes or gt_classes
                    if 'classes' in anno or 'gt_classes' in anno:
                        category_name = anno['classes'] if 'classes' in anno \
                                        else anno['gt_classes']
                        category_id = [cat['id'] for cat in self._categories
                                       if cat['name'] == category_name]
                        if not category_id:
                            cur_ids = [cat['id'] for cat in self._categories]
                            category_id = max(cur_ids) + 1 if cur_ids else 1
                            self._categories.append({
                                'id': category_id,
                                'name': category_name})
                        else:
                            category_id = category_id[0]
                        anno.update({'category_id': category_id})
                    else:
                        pass
            if 'iscrowd' not in anno:
                anno['iscrowd'] = 0.0
            if 'area' not in anno:
                assert 'bbox' in anno
                anno['area'] = anno['bbox'][2] * anno['bbox'][3]
            # if 'keypoints' not in anno:
            #     anno['keypoints'] = [0.0 for _ in range(3 * 17)]
            # if 'num_keypoints' not in anno:
            #     anno['num_keypoints'] = 0   # len(anno['keypoints']) // 3
            self._annotations[self._anno_id] = anno
            self._anno_id += 1

    def set_info(self, **kwargs):
        """
        Add info in coco.
        """
        self._info.update(kwargs)

    def set_categories(self, cats=None):
        """
        Set categories.
        """
        if isinstance(cats, dict):
            self._categories = [cats]
        elif isinstance(cats, list):
            self._categories = cats
        else:
            raise ValueError

    def set_videos(self):
        """
        Set videos based on self._video_name_to_id.

        """
        if hasattr(self, '_videos') and hasattr(self, '_video_name_to_id'):
            self._videos = [{'id': self._video_name_to_id[k], 'name': k} for k in self._video_name_to_id]
        else:
            logging.warning('videos not found')
        # revise begin and end of a video.
        videoToImgs = defaultdict(list)
        for image in self._images.values():
            videoToImgs[image['video_id']].append(image['id'])
        for video_id in videoToImgs:
            imageIds = videoToImgs[video_id]
            prev_imageIds = [self._images[x]['prev_image_id'] for x in imageIds]
            next_imageIds = [self._images[x]['next_image_id'] for x in imageIds]
            for image_id in imageIds:
                if self._images[image_id]['prev_image_id'] not in imageIds:
                    self._images[image_id]['prev_image_id'] = -1
                if self._images[image_id]['next_image_id'] not in imageIds:
                    self._images[image_id]['next_image_id'] = -1

    def save(self, savename):
        """
        Save to json file.
        """
        if hasattr(self, '_videos') and len(self._videos) == 0:
            self.set_videos()
        if len(self._categories) == 0:
            logger.warning('categories is empty, you can add classes or '
                    'gt_classes in roidb to generate correct categories.')
            self.set_categories({'id': 1, 'name': 'default'})
        data = {
            'images': [self._images[x] for x in self._images],
            'annotations': [self._annotations[x] for x in self._annotations],
            'categories': self._categories,
            'info': self._info,
        }
        if hasattr(self, '_actions'):
            data.update({'actions': self._actions})
        if hasattr(self, '_videos'):
            data.update({'videos': self._videos})
        logging.info(f'saving to {savename} ...')
        with open(savename, 'w') as fid:
            json.dump(data, fid)


def save_to_cocoDt(savefile, roidb, **kwargs):
    """Save roidb to coco_dt format.

    Parameters
    ----------
    savefile : str
        filename
    roidb : list
        roidb list

    Returns
    -------
    None

    Example
    -------
    >>> from piata.coco import save_to_cocoDt
    >>> save_to_cocoDt('output.json', roidb)

    """
    assert isinstance(savefile, str)
    assert isinstance(roidb, list)
    roidb_to_cocoDt(roidb, savefile, **kwargs)
    logging.info('saved to %s' % savefile)


def save_to_cocoGt(savefile, roidb, **kwargs):
    """Save roidb to coco_gt format.

    Parameters
    ----------
    savefile : str
        filename
    roidb : list
        roidb list

    Returns
    -------
    None

    Example
    -------
    >>> from piata.coco import save_to_cocoGt
    >>> save_to_cocoGt('output.json', roidb)

    """
    assert isinstance(savefile, str)
    assert isinstance(roidb, list)
    writer = COCOWriter()
    writer.add_roidb(roidb)
    writer.save(savefile)
