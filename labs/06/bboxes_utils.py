#!/usr/bin/env python3
import numpy as np

from svhn_dataset import SVHN

def bbox_area(a):
    return max(0, a[SVHN.BOTTOM] - a[SVHN.TOP]) * max(0, a[SVHN.RIGHT] - a[SVHN.LEFT])

def bbox_iou(a, b):
    """ Compute IoU for two bboxes a, b.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    intersection = [
        max(a[SVHN.TOP], b[SVHN.TOP]),
        max(a[SVHN.LEFT], b[SVHN.LEFT]),
        min(a[SVHN.BOTTOM], b[SVHN.BOTTOM]),
        min(a[SVHN.RIGHT], b[SVHN.RIGHT]),
    ]
    if intersection[SVHN.RIGHT] <= intersection[SVHN.LEFT] or intersection[SVHN.BOTTOM] <= intersection[SVHN.TOP]:
        return 0
    return bbox_area(intersection) / float(bbox_area(a) + bbox_area(b) - bbox_area(intersection))

def bbox_to_fast_rcnn(anchor, bbox):
    """ Convert `bbox` to a Fast-R-CNN-like representation relative to `anchor`.

    The `anchor` and `bbox` are four-tuples (top, left, bottom, right);
    you can use SVNH.{TOP, LEFT, BOTTOM, RIGHT} as indices.

    The resulting representation is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - np.log(bbox_height / anchor_height)
    - np.log(bbox_width / anchor_width)
    """
    assert anchor[SVHN.BOTTOM] > anchor[SVHN.TOP]
    assert anchor[SVHN.RIGHT] > anchor[SVHN.LEFT]
    assert bbox[SVHN.BOTTOM] > bbox[SVHN.TOP]
    assert bbox[SVHN.RIGHT] > bbox[SVHN.LEFT]

    anchor_height = anchor[SVHN.BOTTOM] - anchor[SVHN.TOP]
    anchor_width = anchor[SVHN.RIGHT] - anchor[SVHN.LEFT]
    anchor_y_center = anchor_height / 2. + anchor[SVHN.TOP]
    anchor_x_center = anchor_width / 2. + anchor[SVHN.LEFT]

    bbox_height = bbox[SVHN.BOTTOM] - bbox[SVHN.TOP]
    bbox_width = bbox[SVHN.RIGHT] - bbox[SVHN.LEFT]
    bbox_y_center = bbox_height / 2. + bbox[SVHN.TOP]
    bbox_x_center = bbox_width / 2. + bbox[SVHN.LEFT]

    return ((bbox_y_center - anchor_y_center) / anchor_height,
            (bbox_x_center - anchor_x_center) / anchor_width,
            np.log(bbox_height / anchor_height),
            np.log(bbox_width / anchor_width))

def bbox_from_fast_rcnn(anchor, fast_rcnn):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`."""
    assert anchor[SVHN.BOTTOM] > anchor[SVHN.TOP]
    assert anchor[SVHN.RIGHT] > anchor[SVHN.LEFT]

    anchor_height = anchor[SVHN.BOTTOM] - anchor[SVHN.TOP]
    anchor_width = anchor[SVHN.RIGHT] - anchor[SVHN.LEFT]
    anchor_y_center = anchor_height / 2. + anchor[SVHN.TOP]
    anchor_x_center = anchor_width / 2. + anchor[SVHN.LEFT]

    bbox_height = np.exp(fast_rcnn[2]) * anchor_height
    bbox_width = np.exp(fast_rcnn[3]) * anchor_width

    bbox_y_center = fast_rcnn[0] * anchor_height + anchor_y_center
    bbox_x_center = fast_rcnn[1] * anchor_width + anchor_x_center

    return (bbox_y_center - bbox_height / 2.,
            bbox_x_center - bbox_width / 2.,
            bbox_y_center + bbox_height / 2.,
            bbox_x_center + bbox_width / 2.)

def bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold):
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` as assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if not
      gold object was assigned to the anchor

    Algorithm:
    - First, gold objects are sequentially processed. For each gold object,
      find the first unused anchor with largest IoU and if the IoU is > 0,
      assign the object to the anchor.
    - Second, anchors unassigned so far are sequentially processed. For each
      anchor, find the first gold object with the largest IoU, and if the
      IoU is >= threshold, assign the object to the anchor.
    - considred indeces
    """

    anchor_classes = np.zeros(len(anchors), np.int32)
    anchor_bboxes = np.zeros([len(anchors), 4], np.float32)
    weights = np.ones(len(anchors), np.float32)

    # TODO: Sequentially for each gold object, find the first unused anchor
    # with the largest IoU and if the IoU is > 0, assign the object to the anchor.
    indexed_anchors = list(enumerate(anchors))
    indexed_bboxes = list(enumerate(gold_bboxes))

    for g_bbox_idx, g_bbox in indexed_bboxes[:]:
        if not indexed_anchors:
            break
        anchor_idx, anchor = max(indexed_anchors, key=lambda x: bbox_iou(g_bbox, x[1]))
        if bbox_iou(anchor, g_bbox) > 0:
            indexed_anchors.remove((anchor_idx, anchor))
            anchor_classes[anchor_idx] = gold_classes[g_bbox_idx] + 1
            anchor_bboxes[anchor_idx, :] = bbox_to_fast_rcnn(anchor, g_bbox)


    # TODO: Sequentially for each unassigned anchor, find the first gold object
    # with the largest IoU. If the IoU >= threshold, assign the object to the anchor.

    for anchor_idx, anchor in indexed_anchors:
        if not indexed_bboxes:
            break
        g_bbox_idx, g_bbox = max(indexed_bboxes, key=lambda x: bbox_iou(anchor, x[1]))
        if bbox_iou(anchor, g_bbox) >= iou_threshold:
            anchor_classes[anchor_idx] = gold_classes[g_bbox_idx] + 1
            anchor_bboxes[anchor_idx,:] = bbox_to_fast_rcnn(anchor, g_bbox)
        elif bbox_iou(anchor, g_bbox) > 0.3:
            weights[anchor_idx] = 0.

    return anchor_classes, anchor_bboxes, weights

import unittest
class Tests(unittest.TestCase):
    def test_bbox_to_from_fast_rcnn(self):
        for anchor, bbox, fast_rcnn in [
                [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
                [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
                [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
                [[0, 0, 10, 10], [0, 0, 20, 20], [.5, .5, np.log(2), np.log(2)]],
        ]:
            np.testing.assert_almost_equal(bbox_to_fast_rcnn(anchor, bbox), fast_rcnn, decimal=3)
            np.testing.assert_almost_equal(bbox_from_fast_rcnn(anchor, fast_rcnn), bbox, decimal=3)

    def test_bboxes_training(self):
        anchors = [[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]]
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1/5), np.log(1/5)]], 0.5],
                [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0, 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
        ]:
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)

if __name__ == '__main__':
    unittest.main()
