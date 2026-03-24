import torch
import pandas as pd
import numpy as np
import os
from model.model import ResNet1DCNN

def predict_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './checkpoints/best_model_ResNet1D_tt.pth'
    test_brt_path = './2015_2020_dataset/test_data/test_brt.csv'
    train_brt_path = './2015_2020_dataset/train_data/train_brt.csv'
    train_t_path = './2015_2020_dataset/train_data/train_t.csv'
    output_path = './output/predict_tt.csv'

    print("Calculating normalization statistics...")
    df_train_x = pd.read_csv(train_brt_path, header=None).iloc[:, 8:]
    df_train_y = pd.read_csv(train_t_path, header=None)
    
    stats = {
        'x_mean': df_train_x.values.mean(axis=0).astype(np.float32),
        'x_std': df_train_x.values.std(axis=0).astype(np.float32),
        'y_mean': df_train_y.values.mean(axis=0).astype(np.float32),
        'y_std': df_train_y.values.std(axis=0).astype(np.float32)
    }

    model = ResNet1DCNN(input_channels=1, input_len=8, output_dim=83)
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    df_test_x = pd.read_csv(test_brt_path, header=None)
    v_band_test = df_test_x.iloc[:, 8:].values.astype(np.float32) 
    
    v_band_norm = (v_band_test - stats['x_mean']) / (stats['x_std'] + 1e-6)
    input_tensor = torch.from_numpy(v_band_norm).float().to(device)
    
    print("Starting prediction...")
    with torch.no_grad():
        output_norm = model(input_tensor)
        output_norm = output_norm.cpu().numpy()

    predict_raw = output_norm * (stats['y_std'] + 1e-6) + stats['y_mean']
    df_predict = pd.DataFrame(predict_raw).round(2)
    if not os.path.exists('./output'): os.makedirs('./output')
    
    df_predict.to_csv(output_path, index=False, header=None)
    print(f"Done! Results saved to: {output_path}")

if __name__ == "__main__":
    predict_process()