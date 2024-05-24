import torch
from torch.utils.data import Dataset
import numpy as np 
import albumentations as A
import cv2
from torchvision.transforms import ToTensor

DIR_TRAIN = 'input/train_powdery'

class LeafDataset(Dataset):
    def __init__(self,image_ids,dataframe,transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def __getitem__(self,index):
        index_df = self.df.iloc[index]
        image_id = index_df['image_id']
        records = self.df[self.df['image_id'] == image_id]

        source = records['source'].values[0]
        image = cv2.imread(f'{DIR_TRAIN}/{source}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        #Obtaing boxes values
        boxes = records[['x', 'y', 'w', 'h']].values
        #Area of bb
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = records['label'].tolist()
        try:
            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                }
                sample['image'] = ToTensor()(sample['image'])
                image = sample['image']
                boxes = sample['bboxes']
                labels = sample['labels']
        except Exception as e:
            print(source)
            print(str(e))
            
        #Normalizing BBOXES
        _,h,w = image.shape
        try:
            boxes = A.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        except Exception as e:
            print(str(e))
            print(source)
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, image_id