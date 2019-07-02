

from utils.iou import get_iou_matrix


def test_iou():
    boxes_pred = [[1, 100, 1, 200]]
    boxes_gt = [[100., 120., 150., 160.],[12, 110, 130., 160.]]
    iou_matrix = get_iou_matrix(boxes_pred, boxes_gt)
    assert iou_matrix.shape == (len(boxes_pred), len(boxes_gt))

