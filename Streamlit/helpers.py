import numpy as np
import cv2
import torch

def calculate_score(game_state):
    game_state = np.array(game_state, dtype=int)
    max_score = -1
    
    if np.all(game_state > 0):
        new_state = game_state - 1
        score_sub_tree = calculate_score(tuple(new_state))
        score = (1500 + score_sub_tree) if score_sub_tree > -1 else 0
        max_score = max(max_score, score)

    if np.all(game_state[1:] > 0):
        new_state = game_state.copy()
        new_state[1:] -= 1
        score_sub_tree = calculate_score(tuple(new_state))
        score = (750 + score_sub_tree) if score_sub_tree > -1 else 0
        max_score = max(max_score, score)

    if np.all(game_state[:-1] > 0):
        new_state = game_state.copy()
        new_state[:-1] -= 1
        score_sub_tree = calculate_score(tuple(new_state))
        score = (500 + score_sub_tree) if score_sub_tree > -1 else 0
        max_score = max(max_score, score)

    score = 0
    for i, count in enumerate(game_state):
        if count >= 3:
            base = 1000 if i == 0 else (i + 1) * 100
            score += base * (1 << (count - 3)) 
            game_state[i] = 0
        elif i == 0:
            score += 100 * count
            game_state[i] = 0
        elif i == 4:
            score += 50 * count
            game_state[i] = 0
    if np.sum(game_state) == 0:
        max_score = max(max_score, score)
    return max_score

def filter_and_crop_objects(result, image, iou_threshold=0.5):
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    idxs = np.argsort(-areas)  # Descending by area
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        rest = idxs[1:]
        if len(rest) == 0:
            break

        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[current] + areas[rest] - inter_area
        iou = inter_area / (union_area + 1e-6)

        idxs = rest[iou < iou_threshold]

    # Crop boxes from image
    cropped_images = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        cropped = image[y1:y2, x1:x2]
        cropped_images.append(cropped)

    return cropped_images


def run_inference_and_show(crops, model, device, size=(128,128)):
    # Preprocess crops → tensor batch
    tensors = []
    for crop in crops:
        resized = cv2.resize(crop, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Random horizontal flip
        rgb = cv2.flip(rgb, 1)  # 1 = horizontal

        rgb = cv2.flip(rgb, 0) 
        tensor = torch.tensor(rgb).permute(2,0,1).float() / 255.0  # [C,H,W], normalize 0-1
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)

    # Run inference
    with torch.no_grad():
        preds = model(batch)
        pred_ids = torch.argmax(preds, dim=1).cpu()
        print(pred_ids + 1)