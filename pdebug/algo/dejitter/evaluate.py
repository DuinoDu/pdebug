"""
Jitter evaluation.
"""
import numpy as np


class Evaluate(object):

    """
    Evaluate jitter.

    Refer to https://arxiv.org/pdf/1611.06467.pdf

    Examplee
    --------
    >>> boxes_list = [[XYXY], ...]
    >>> m_boxes_center, gt_boxes = JitterEvaluate.boxes_center(boxes_list)
    >>> m_boxes_scale = JitterEvaluate.boxes_scale_ratio(boxes_list)
    >>> kps_list = [[XYV], ...]
    >>> m_kps_position = JitterEvaluate.keypoints(kps_list, gt_boxes=gt_boxes)

    """

    @staticmethod
    def boxes_center(boxes_list, gt_boxes=None):
        """
        Center Position Error.

        Parameter
        ---------
        boxes_list : boxes list
            XYXY
        """
        boxes_list = np.asarray(boxes_list, dtype=np.float32)
        assert boxes_list.ndim == 2

        if gt_boxes is not None:
            raise NotImplementedError
        else:
            mean_bbox = np.mean(boxes_list, axis=0)
            gt_boxes = np.repeat(
                mean_bbox[np.newaxis, :], boxes_list.shape[0], axis=0
            )

        pred_center_x = (boxes_list[:, 0] + boxes_list[:, 2]) / 2
        pred_center_y = (boxes_list[:, 1] + boxes_list[:, 3]) / 2
        gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        width_g = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
        height_g = gt_boxes[:, 3] - gt_boxes[:, 1] + 1

        error_x = (pred_center_x - gt_center_x) / width_g
        error_y = (pred_center_y - gt_center_y) / height_g
        error = np.std(error_x) + np.std(error_y)
        print("Center Position Error: %.2f%%" % (error * 100))
        return error, gt_boxes

    @staticmethod
    def boxes_scale_ratio(boxes_list, gt_boxes=None):
        """
        Scale and Ratio Error.

        Parameter
        ---------
        boxes_list : boxes list
            XYXY
        """
        boxes_list = np.asarray(boxes_list, dtype=np.float32)
        assert boxes_list.ndim == 2

        if gt_boxes is not None:
            raise NotImplementedError
        else:
            mean_bbox = np.mean(boxes_list, axis=0)
            gt_boxes = np.repeat(
                mean_bbox[np.newaxis, :], boxes_list.shape[0], axis=0
            )

        width_g = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
        height_g = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
        width_p = boxes_list[:, 2] - boxes_list[:, 0] + 1
        height_p = boxes_list[:, 3] - boxes_list[:, 1] + 1

        error_scale = np.sqrt((width_p * height_p) / (width_g * height_g))
        error_ratio = (width_p / height_p) / (width_g / height_g)
        error = np.std(error_scale) + np.std(error_ratio)
        print("Scale and Ratio Error: %.2f%%" % (error * 100))
        return error

    @staticmethod
    def keypoints(kps_list, gt_kps=None, gt_boxes=None):
        """
        Keypoints Position Error.

        Parameter
        ---------
        kps_list : keypoints list
            XYV
        """
        kps_list = np.asarray(kps_list, dtype=np.float32)
        assert kps_list.ndim == 2

        if gt_kps is not None:
            raise NotImplementedError
        else:
            mean_kps = np.mean(kps_list, axis=0)
            gt_kps = np.repeat(
                mean_kps[np.newaxis, :], kps_list.shape[0], axis=0
            )

        pred_x = kps_list[:, 0::3]
        pred_y = kps_list[:, 1::3]
        gt_x = gt_kps[:, 0::3]
        gt_y = gt_kps[:, 1::3]

        error_x = pred_x - gt_x
        error_y = pred_y - gt_y

        if gt_boxes is not None:
            width_g = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
            height_g = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
            num_kps = pred_x.shape[1]
            width_g = width_g[:, np.newaxis].repeat(num_kps, axis=1)
            height_g = height_g[:, np.newaxis].repeat(num_kps, axis=1)
            error_x /= width_g
            error_y /= height_g
        error = np.std(error_x, axis=0) + np.std(error_y, axis=0)
        print("Center Position Error:")
        for err in error:
            print("  %.2f%%" % (err * 100))
        return error.mean()
