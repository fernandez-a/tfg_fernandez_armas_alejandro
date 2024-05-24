from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def prepare_data(input_path):
    marking = pd.read_csv(input_path)
    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',').astype(int)))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)
    unique_image_ids = marking["image_id"].unique()
    

    df_train, df_test = train_test_split(unique_image_ids, test_size=0.2, random_state=42)
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)

    df_train = marking[marking["image_id"].isin(df_train)]
    df_valid = marking[marking["image_id"].isin(df_valid)]


    print(f"Train: {df_train.shape}, Valid: {df_valid.shape}")
    return marking, df_train, df_valid, df_test


def prepare_test_data(input_path):
    df_test = pd.read_csv(input_path)
    bboxs = np.stack(df_test['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',').astype(int)))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df_test[column] = bboxs[:,i]
    df_test.drop(columns=['bbox'], inplace=True)
    print(f"Test: {df_test.shape}")
    return df_test