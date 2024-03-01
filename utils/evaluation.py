from shapely.geometry import box
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils.leafDataset import LeafDataset
import albumentations as A
import numpy as np
from utils.transformation import get_valid_transforms
from utils.train_eval import collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2


def calculate_iou(gt_bbox, pred_bbox):
    gt_box = box(gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
    pred_box = box(pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3])
     
    intersection = gt_box.intersection(pred_box).area
    union = gt_box.union(pred_box).area
 
    iou = intersection / union
    return iou

def mean_ious(df_valid, model, device,marking, BATCH_SIZE):
    iou_threshold=0.5
    valid_dataset = LeafDataset(image_ids=df_valid.index.values,
                                 dataframe=marking,
                                 transforms=get_valid_transforms()
                                )
     
    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)
    
    mean_ious = []
    processed_image_ids = []
    true_positives_list = []
    false_positives_list = []
    false_negatives_list = []
    
    for i, (images, targets, image_ids) in enumerate(tqdm(valid_data_loader)):
        _, h, w = images[0].shape  # for de-normalizing images
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        boxes = targets[0]['boxes'].cpu().numpy()
        boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes, h, w)]
        labels = targets[0]['labels'].cpu().numpy()
                
        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")
        
        with torch.no_grad():
            outputs = model(images)
            
        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
        oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
        oboxes = [np.array(box).astype(np.int32) for box in  A.augmentations.bbox_utils.denormalize_bboxes(oboxes, h, w)]
        prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
        
        ious_all = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for box, p, label in zip(oboxes, prob, labels):
            if p > 0.5:
                ious = [calculate_iou(box, gt_box) for gt_box in boxes]
                max_iou = max(ious) if ious else 0 
                ious_all.append(max_iou)
                
                if max_iou >= iou_threshold:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if any(calculate_iou(box, gt_box) > iou_threshold for gt_box in boxes):
                    false_negatives += 1
        
        mean_iou = np.mean(ious_all)
        mean_ious.append(mean_iou)
        processed_image_ids.append(image_ids[0])
        true_positives_list.append(true_positives)
        false_positives_list.append(false_positives)
        false_negatives_list.append(false_negatives)
    
    df = pd.DataFrame({
        'image_id': processed_image_ids,
        'mean_iou': mean_ious,
        'true_positives': true_positives_list,
        'false_positives': false_positives_list,
        'false_negatives': false_negatives_list
    })
    return df

def view_sample(df_valid, model, device, marking, BATCH_SIZE):
    valid_dataset = LeafDataset(image_ids=df_valid.index.values,
                                 dataframe=marking,
                                 transforms=get_valid_transforms()
                                )
     
    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)
    
    images, targets, image_ids = next(iter(valid_data_loader))
    _, h, w = images[0].shape  # for de-normalizing images
    
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    boxes = targets[0]['boxes'].cpu().numpy()
    boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes, h, w)]
    
    sample = images[0].permute(1, 2, 0).cpu().numpy()
    
    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")
    
    with torch.no_grad():
        outputs = model(images)
        
    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2]+box[0], box[3]+box[1]),
                  (220, 0, 0), 1)
    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in  A.augmentations.bbox_utils.denormalize_bboxes(oboxes, h, w)]
    prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
    
    for box,p in zip(oboxes,prob):
        
        if p >0.5:
            color = (0,0,220)
            cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2]+box[0], box[3]+box[1]),
                  color, 1)
    
    ax.set_axis_off()
    ax.imshow(sample)

    plt.show()