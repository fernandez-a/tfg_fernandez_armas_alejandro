import torch
from utils.evaluation import mean_ious, view_sample
from model import DETRModel
from utils.prepare_data import prepare_data
import matplotlib.pyplot as plt
def main():
    num_classes = 3
    num_queries = 100
    BATCH_SIZE = 8
    marking, df_train, df_valid = prepare_data('input/train.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    model.load_state_dict(torch.load("./model_weights.pth"))
    view_sample(df_valid=df_valid, model=model, device=device,marking=marking, BATCH_SIZE=BATCH_SIZE)
    mean_ious(df_valid=df_valid, model=model, device=torch.device('cuda'),marking=marking, BATCH_SIZE=BATCH_SIZE)

if __name__ == "__main__":
    main()