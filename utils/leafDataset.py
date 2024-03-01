import torch
from torch.utils.data import Dataset
import numpy as np 
import albumentations as A
import cv2
DIR_TRAIN = 'input/train'

class LeafDataset(Dataset):
    def __init__(self,image_ids,dataframe,transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def __getitem__(self,index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        source = records['source'].values[0]
        image = cv2.imread(f'{DIR_TRAIN}/{source}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # DETR takes in data in coco format 
        boxes = records[['x', 'y', 'w', 'h']].values
        
        #Area of bb
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.zeros(len(boxes), dtype=np.int32)

        try:
            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']
                boxes = sample['bboxes']
                labels = sample['labels']
        except:
            print(source)
            
        #Normalizing BBOXES
            
        _,h,w = image.shape
        try:
            boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        except:
            print(source)
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, image_id