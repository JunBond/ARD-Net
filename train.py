import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy 
import time 
import os
import random
import logging  
from model.model import ResNet1DCNN 

def setup_experiment(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not os.path.exists('./logs'): os.makedirs('./logs')
    log_filename = f"./logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])
    logging.info(f"Seed set to {seed}")
    logging.info(f"Log file saved at {log_filename}")
    return logging

class CSVDataset(Dataset):
    def __init__(self, brt_path, t_path, is_train=True, stats=None):
        df_x = pd.read_csv(brt_path, header=None)
        df_y = pd.read_csv(t_path, header=None)
        
        # 只取后8个通道 (V接收机)
        x_raw = df_x.iloc[:, 8:].values.astype(np.float32)
        y_raw = df_y.values.astype(np.float32)
        
        # --- 数据归一化 (Z-Score) ---
        if is_train:
            self.stats = {
                'x_mean': x_raw.mean(axis=0), 'x_std': x_raw.std(axis=0),
                'y_mean': y_raw.mean(axis=0), 'y_std': y_raw.std(axis=0)
            }
        else:
            self.stats = stats

        # 归一化处理
        x_norm = (x_raw - self.stats['x_mean']) / (self.stats['x_std'] + 1e-6)
        y_norm = (y_raw - self.stats['y_mean']) / (self.stats['y_std'] + 1e-6)
        
        self.x_data = torch.from_numpy(x_norm)
        self.y_data = torch.from_numpy(y_norm)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

def load_data_process(batch_size=64):
    train_brt_path = './2015_2020_dataset/train_data/train_brt.csv'
    train_t_path = './2015_2020_dataset/train_data/train_t.csv'
    test_brt_path = './2015_2020_dataset/test_data/test_brt.csv'
    test_t_path = './2015_2020_dataset/test_data/test_t.csv'
    
    train_dataset = CSVDataset(train_brt_path, train_t_path, is_train=True)
    test_dataset = CSVDataset(test_brt_path, test_t_path, is_train=False, stats=train_dataset.stats)
    
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Dataset Loaded: Train={len(train_dataset)}, Test={len(test_dataset)}")
    return train_dataloader, val_dataloader, train_dataset.stats

def train_model_process(model, train_dataloader, val_dataloader, num_epochs, lr=0.0003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    best_loss = float('inf')
    train_loss_all, val_loss_all = [], []
    since = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_num = 0.0, 0
        
        for b_x, b_y in train_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model(b_x)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        model.eval()
        val_loss, val_num = 0.0, 0
        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = model(b_x)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)

        avg_train_loss = train_loss / train_num
        avg_val_loss = val_loss / val_num
        train_loss_all.append(avg_train_loss)
        val_loss_all.append(avg_val_loss)

        logging.info(f"Epoch {epoch}/{num_epochs-1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if not os.path.exists('./checkpoints'): os.makedirs('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/best_model_ResNet1D_tt.pth')
            logging.info(f"*** Best Model Saved (Loss: {best_loss:.6f}) ***")

    time_elapsed = time.time() - since
    logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return pd.DataFrame({"epoch": range(num_epochs), "train_loss_all": train_loss_all, "val_loss_all": val_loss_all})

def matplot_acc_loss(train_process):
    plt.style.use('ggplot') 
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(train_process['epoch'], train_process.train_loss_all, label='Train Loss (Normalized)')
    plt.plot(train_process['epoch'], train_process.val_loss_all, label='Val Loss (Normalized)')
    plt.title('Training Process (Learning Rate: 0.0003)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Normalized)')
    plt.legend()
    if not os.path.exists('./output'): os.makedirs('./output')
    plt.savefig('./output/normalized_loss_curve_tt.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    logger = setup_experiment(seed=42)
    Net = ResNet1DCNN(input_channels=1, input_len=8, output_dim=83)
    train_loader, val_loader, data_stats = load_data_process(batch_size=32)
    process_record = train_model_process(Net, train_loader, val_loader, num_epochs=100, lr=0.0003)
    
    matplot_acc_loss(process_record)
    logging.info("Training process finished.")