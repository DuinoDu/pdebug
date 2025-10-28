# -*- coding: utf-8 -*-
"""target visualization tool.
"""

import copy
import os
import time

from pdebug.visp.geomentry import Line

import cv2
import numpy as np

try:
    import torch
except ImportError as e:
    torch = None
try:
    import mxnet as mx
except ImportError as e:
    mx = None


__all__ = [
    "minmax_to_0_255",
    "unnormalize",
    "heatmap",
    "dump_batch_and_heatmap",
]


def minmax_to_0_255(data, axis=(2, 3)):
    """convert min,max to 0,255"""
    new_data = np.zeros_like(data)
    max_value = np.amax(data, axis=axis, keepdims=True)
    min_value = np.amin(data, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        data = np.true_divide(
            np.subtract(data, min_value), np.subtract(max_value, min_value)
        )
        data = np.nan_to_num(data)
    return data * 255


def unnormalize(
    data,
    mean=(128, 128, 128),
    std=(1.0 / 128, 1.0 / 128, 1.0 / 128),
    isrgb=False,
    isyuv=False,
    ishwc=False,
    normal_method=None,
):
    """Unnormalize image data to 0~255.

    Parameters
    ----------
    data : np.array, [B, 3, H, W]
        input tensor data.
    mean : tuple or float
        normalize mean.
    std : tuple or float
        normalize std.
    isrgb : bool
        convert rgb to bgr.
    isyuv : bool
        convert yuv to bgr.
    ishwc : bool
        default False, convert chw to hwc.

    Returns
    -------
    np.array
        [B, H, W, 3] or [H, W, 3], in cv2_bgr order

    """
    if torch is not None and isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        _normal_method = "pytorch"
    else:
        _normal_method = "mxnet"
    if not normal_method:
        normal_method = _normal_method

    if isinstance(data, list):
        data = np.asarray(data)

    # make a copy to avoid change input data.
    data = copy.deepcopy(data)

    assert isinstance(data, np.ndarray), "input data should be np.array type"
    if isinstance(mean, float) or isinstance(mean, int):
        mean = (mean, mean, mean)
    if isinstance(std, float) or isinstance(std, int):
        std = (std, std, std)

    ndim = data.ndim
    if ndim == 3:
        new_data = data[np.newaxis]
    else:
        new_data = data

    if not ishwc:
        new_data = np.transpose(new_data, (0, 2, 3, 1))  # BCHW -> BHWC
    batch_size = new_data.shape[0]
    data_h = new_data.shape[1]
    data_w = new_data.shape[2]
    data_c = new_data.shape[3]
    assert data_h > data_c and data_w > data_c
    if data_c == 1:
        new_data = np.concatenate((new_data, new_data, new_data), axis=3)

    if np.abs(mean[0]) < 1.0:
        unnormal_inside_255 = True
    else:
        unnormal_inside_255 = False
    for i in range(3):
        if normal_method == "pytorch":
            if unnormal_inside_255:
                new_data[:, :, :, i] = (
                    new_data[:, :, :, i] * std[i] + mean[i]
                ) * 255.0
            else:
                new_data[:, :, :, i] = (
                    new_data[:, :, :, i] * 255.0 * std[i] + mean[i]
                )
        else:
            new_data[:, :, :, i] = new_data[:, :, :, i] / std[i] + mean[i]
    new_data = new_data.astype(np.uint8)

    if isrgb:
        for i in range(batch_size):
            new_data[i] = cv2.cvtColor(new_data[i], cv2.COLOR_RGB2BGR)
    elif isyuv:
        for i in range(batch_size):
            new_data[i] = cv2.cvtColor(new_data[i], cv2.COLOR_YUV2BGR)

    if ndim == 3:
        new_data = new_data[0]
    return new_data


def heatmap(
    data,
    image=None,
    merge=False,
    drawscore=False,
    save=None,
    extra_label=None,
    no_text=False,
):
    """Transform heatmap(0~1) to 0~255.

    Parameters
    ----------
    data : np.array
        Input data, [B, C, H, W] or [C, H, W]
    image : np.array
        If given, draw heatmap on image, [B, H, W, 3] or [C, H, W]
    merge : bool
        Merge data'channel to one channel.
    drawscore : bool
        draw max score
    save : bool
        save result to jpg
    extra_label : np.array or list of np.array
        draw extra label, shape is [B, C]
    no_text : bool
        disable text show on heatmap

    Returns
    -------
    list, or 4D-tensor
        heatmap in cv2image format

    """
    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if mx is not None and isinstance(data, mx.ndarray.NDArray):
        data = data.asnumpy()

    ndim = data.ndim
    if ndim == 3:
        data = data[np.newaxis]
    if drawscore:
        # [B, C]
        max_scores = data.reshape(data.shape[0], data.shape[1], -1).max(axis=2)
    if extra_label is not None:
        if isinstance(extra_label, np.ndarray):
            extra_label = [extra_label]
        for label_i in extra_label:
            assert label_i.ndim == 2

    data = minmax_to_0_255(data, axis=(2, 3))
    data = np.transpose(data, (0, 2, 3, 1))  # to bhwc
    batch_size, h, w, ch = data.shape
    data = data.astype(np.uint8)

    if image is not None:
        if isinstance(image, list):
            image = np.asarray(image)
        if hasattr(image, "numpy"):
            image = image.numpy()
        image_ = image.copy()
        if image_.ndim == 3:
            image_ = image_[np.newaxis]
        assert data.shape[0] == image_.shape[0]
        img_h = image_.shape[1]
        img_w = image_.shape[2]
        data_resize = np.zeros((batch_size, img_h, img_w, ch))
        for i in range(batch_size):
            for j in range(ch):
                data_resize[i, :, :, j] = cv2.resize(
                    data[i, :, :, j], (img_w, img_h)
                )
        data = data_resize
        batch_size, h, w, ch = data.shape

    if merge:
        res = np.zeros((batch_size, h, w, 3), dtype=np.uint8)
        data = np.sum(data, axis=3).astype(np.int32)
        data[data > 255] = 255
    else:
        res = np.zeros((batch_size, h, w * ch, 3), dtype=np.uint8)
        multi = np.concatenate(
            [data[:, :, :, x] for x in range(ch)], axis=2
        )  # [b, h, w*ch]
        data = multi
        if image is not None:
            image_ = np.concatenate(
                [image_ for _ in range(ch)], axis=2
            )  # [b, h, w*ch, 3]

    if image is not None:
        for batch_ind in range(batch_size):
            this_image = image_[batch_ind]  # [H, W, 3]
            this_data = data[batch_ind]  # [h, w]
            this_data = minmax_to_0_255(this_data, axis=(0, 1))
            this_data_color = cv2.applyColorMap(
                this_data.astype(np.uint8), cv2.COLORMAP_JET
            )
            masked_image = this_data_color * 0.6 + this_image * 0.7
            res[batch_ind] = masked_image.astype(np.uint8)
            if not merge:
                if no_text or this_image.shape[0] < 50:
                    continue
                # draw text
                for i in range(ch):
                    res_seg = res[batch_ind, :, i * w : (i + 1) * w, :]
                    str_draw = str(i)
                    if drawscore:
                        str_draw += " / %.2f" % max_scores[batch_ind, i]
                    res[batch_ind, :, i * w : (i + 1) * w, :] = cv2.putText(
                        res_seg,
                        str_draw,
                        (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                    if extra_label is not None:
                        str_draw = ""
                        for each_label in extra_label:
                            str_draw += "%.2f / " % each_label[batch_ind, i]
                        str_draw = str_draw[:-3]
                    res[batch_ind, :, i * w : (i + 1) * w, :] = cv2.putText(
                        res_seg,
                        str_draw,
                        (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
    else:
        res[:, :, :, 0] = data[:, :, :]
        res[:, :, :, 1] = data[:, :, :]
        res[:, :, :, 2] = data[:, :, :]

    if save:
        if len(res) > 1:
            cv2.imwrite(save, np.concatenate(res, axis=0))
        else:
            cv2.imwrite(save, res[0])

    if ndim == 3:
        res = res[0].astype(np.uint8)
    return res


def dump_batch_and_heatmap(data, config, output="cache"):
    """
    dump vis of batch and heatmap to file.

    Parameters
    ----------
    data : list
        [batch, kps_label_pred, kps_label, kps_label_loss]
        [batch, kps_label_pred, kps_label, kps_label_loss, paf_pred, paf, paf_loss]
    config : dict
        gluonperson config dict
    output : str
        output directory

    """
    if not os.path.exists(output):
        os.makedirs(output)

    batch = data[0]
    imgs = unnormalize(
        batch.asnumpy(),
        mean=config.network.input_mean,
        std=config.network.input_scale,
        isrgb=True,
    )

    kps_label_pred = data[1]
    kps_label = data[2]
    kps_label_loss = data[3]
    # vis kps heatmap
    preds = heatmap(
        kps_label_pred.asnumpy(), image=imgs, merge=False, drawscore=True
    )
    labels = heatmap(kps_label.asnumpy(), image=imgs, merge=False)

    has_paf = False
    score_thres = 0.2
    paf_thres = 0.1
    if len(data) > 4:
        has_paf = True
        paf_pred = data[4]
        paf = data[5]
        paf_loss = data[6]
        # vis paf heatmap
        preds_paf = heatmap(
            paf_pred.asnumpy(), image=imgs, merge=False, drawscore=True
        )
        labels_paf = heatmap(paf.asnumpy(), image=imgs, merge=False)

        # vis line
        lines_heatmap = []
        lines_paf = []
        for ind in range(len(preds)):
            assert kps_label_pred.shape[1] == 2
            max_ind_0, score_0 = argmax2d(
                kps_label_pred[ind][0].asnumpy(), with_max=True
            )
            max_ind_1, score_1 = argmax2d(
                kps_label_pred[ind][1].asnumpy(), with_max=True
            )
            # get line from heatmap
            if score_0 > score_thres and score_1 > score_thres:
                line_heatmap = Line(p0=max_ind_0, p1=max_ind_1, scale=16.0)
            else:
                line_heatmap = None
            # get line from point and paf
            if score_0 > score_thres or score_1 > score_thres:
                if score_0 > score_1:
                    valid_point = max_ind_0
                else:
                    valid_point = max_ind_1
            else:
                valid_point = None
            paf_mod = np.power(paf_pred[ind][0].asnumpy(), 2) + np.power(
                paf_pred[ind][1].asnumpy(), 2
            )
            paf_x = paf_pred[ind][0].asnumpy()[paf_mod > paf_thres]
            paf_y = paf_pred[ind][1].asnumpy()[paf_mod > paf_thres]
            if valid_point is None or paf_x.shape[0] == 0:
                line_paf = None
            else:
                paf_x = paf_x.mean()
                paf_y = paf_y.mean()
                line_paf = Line(
                    p0=valid_point, unit_vec=(paf_x, paf_y), scale=16.0
                )

            lines_heatmap.append(line_heatmap)
            lines_paf.append(line_paf)

    has_multibin = False
    if len(data) > 7:
        has_multibin = True
        bin_pred = data[7]
        res_pred = data[8]
        angle_pred = data[9]
        bin_label = data[10]
        res_label = data[11]
        angle_label = data[12]

    # save to file
    for ind in range(len(preds)):
        _save = np.concatenate((preds[ind], labels[ind]), axis=0)
        loss_value = kps_label_loss.reshape(-1)[ind].asscalar()
        loss_str = "loss_%.6f" % loss_value
        if has_paf:
            _save_paf = np.concatenate(
                (preds_paf[ind], labels_paf[ind]), axis=0
            )
            _save = np.concatenate((_save, _save_paf), axis=1)
            loss_value = paf_loss.reshape(-1)[ind].asscalar()
            loss_str += "_loss_%.6f" % loss_value
            l1, l2 = lines_heatmap[ind], lines_paf[ind]
            if l1:
                _save_line = l1.draw(imgs[ind], color=(0, 255, 255))
            else:
                _save_line = imgs[ind]
            if l2:
                _save_line = l2.draw(_save_line, color=(0, 0, 255))
            _save_line = cv2.putText(
                _save_line.copy(),
                "heatmap",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            _save_line = cv2.putText(
                _save_line.copy(),
                "paf",
                (10, 70),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # for lint_gt, TODO
            _save_line = np.concatenate((_save_line, imgs[ind]), axis=0)

            # [heatmap, paf, line]
            _save = np.concatenate((_save, _save_line), axis=1)
        if has_multibin:
            _pad_img = np.zeros_like(imgs[ind])
            _pad_img = np.concatenate((imgs[ind], _pad_img), axis=1)
            _pred_img = _pad_img.copy()

            # if bin_pred[ind][0] > 1 or bin_pred[ind][1] > 1:
            #    bin_pred[ind] = softmax(bin_pred[ind])
            _pred_img = cv2.putText(
                _pred_img,
                "bin: %.1f, %.1f" % (bin_pred[ind][0], bin_pred[ind][1]),
                (10, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _pred_img = cv2.putText(
                _pred_img,
                "res: %.3f, %.3f" % (res_pred[ind][0], res_pred[ind][1]),
                (10, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _pred_img = cv2.putText(
                _pred_img,
                "angle: %f" % angle_pred[ind],
                (10, 140),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            _label_img = _pad_img.copy()
            _label_img = cv2.putText(
                _label_img,
                "bin: %d, %d" % (bin_label[ind][0], bin_label[ind][1]),
                (10, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            _label_img = cv2.putText(
                _label_img,
                "res: %.3f, %.3f" % (res_label[ind][0], res_label[ind][1]),
                (10, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            _label_img = cv2.putText(
                _label_img,
                "angle: %f" % angle_label[ind],
                (10, 140),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            _save_multibin = np.concatenate((_pred_img, _label_img), axis=0)
            _save = np.concatenate((_save, _save_multibin), axis=1)

        savefile = os.path.join(
            output, "%.3f_image_%03d_%s.jpg" % (time.time(), ind, loss_str)
        )
        cv2.imwrite(savefile, _save)


def argmax2d(_input, with_max=False):
    index1d = np.argmax(_input)
    index_x, index_y = np.unravel_index(index1d, _input.shape)
    # bring the indices into the right shape
    index = np.array((index_x, index_y))
    if with_max:
        return index, np.max(_input)
    else:
        return index


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)
