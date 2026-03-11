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

import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """
    Applies Non-Max Suppression to filter overlapping bounding boxes.
    
    Args:
        boxes: List or array of boxes in [x1, y1, x2, y2] format.
        scores: List or array of confidence scores for each box.
        iou_threshold: Float. Boxes with IoU > threshold are suppressed.
        
    Returns:
        keep: A list of indices corresponding to the boxes to keep.
    """
    # Convert inputs to numpy arrays for vectorization
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # If no boxes are provided, return an empty list
    if len(boxes) == 0:
        return []

    # Extract coordinates for readability
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of all boxes in the array upfront
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort boxes by their confidence scores in DESCENDING order
    # argsort() returns the indices that would sort the array
    order = scores.argsort()[::-1] 

    keep = []

    # Loop until we have processed all boxes in the 'order' array
    while order.size > 0:
        # Grab the index of the box with the highest remaining score
        i = order[0]
        keep.append(i) # We keep this box!

        # Now, calculate the IoU between this winning box 'i' 
        # and ALL the remaining boxes in the 'order' array simultaneously
        
        # 1. Find the intersection coordinates (vectorized)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 2. Calculate intersection area (vectorized)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # 3. Calculate IoU (vectorized)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        # 4. Suppress overlapping boxes
        # We only want to KEEP boxes where the IoU is LESS than or equal to our threshold
        inds = np.where(iou <= iou_threshold)[0]

        # 5. Update the 'order' array for the next iteration
        # We add 1 to 'inds' because we sliced order[1:] earlier to skip the winning box 'i'
        order = order[inds + 1]

    return keep