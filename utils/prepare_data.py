from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def prepare_data(input_path):
    marking = pd.read_csv(input_path)
    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',').astype(int)))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)

    df_train, df_valid = train_test_split(marking, test_size=0.2, random_state=42)

    df_train.set_index('image_id', inplace=True)
    df_valid.set_index('image_id', inplace=True)
    print(f"Train: {df_train.shape}, Valid: {df_valid.shape}")
    return marking, df_train, df_valid