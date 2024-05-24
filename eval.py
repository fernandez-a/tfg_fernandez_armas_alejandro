import torch
from utils.evaluation import eval_test, view_sample
from utils.model import DETRModel
from utils.prepare_data import prepare_data, prepare_test_data
import matplotlib.pyplot as plt


def main():
    num_classes = 2
    num_queries = 100
    BATCH_SIZE = 8

    df_test = prepare_test_data('./input/test_powdery_final.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    model.load_state_dict(torch.load("./models/detr_powdery_aug_500_box_loss.pth"))
    
    view_sample(df_test=df_test, model=model, device=device,marking=df_test, BATCH_SIZE=BATCH_SIZE)
    df = eval_test(df_test=df_test, model=model, device=torch.device('cuda'),marking=df_test, BATCH_SIZE=BATCH_SIZE)
if __name__ == "__main__":
    main()