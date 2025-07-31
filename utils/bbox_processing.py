def coordinates_converter(bboxes, conversion=None):
    if conversion == "centroids2corners":
        boxes_xy = bboxes[..., 0:2] - 0.5 * bboxes[..., 2:4]
        boxes_wh = bboxes[..., 0:2] + 0.5 * bboxes[..., 2:4]
        bboxes[..., 0:2] = boxes_xy
        bboxes[..., 2:4] = boxes_wh
    elif conversion == "corners2centroids":
        boxes_xy  = (bboxes[..., 0:2] + bboxes[..., 2:4]) // 2
        boxes_wh = bboxes[..., 2:4] - bboxes[..., 0:2]
        bboxes[..., 0:2] = boxes_xy
        bboxes[..., 2:4] = boxes_wh
    return bboxes