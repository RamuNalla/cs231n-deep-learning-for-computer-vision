def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Assumes boxes are formatted as [x1, y1, x2, y2].
    """
    # 1. Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. Calculate the area of the intersection
    # We use max(0, ...) to handle the case where the boxes do not overlap at all
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # 3. Calculate the area of both bounding boxes separately
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 4. Calculate the Union area 
    # (Area A + Area B - Intersection Area to avoid double counting)
    union_area = float(boxA_area + boxB_area - inter_area)

    # 5. Compute the IoU ratio
    # Add a small epsilon to the denominator to prevent division by zero
    iou = inter_area / (union_area + 1e-6)

    return iou