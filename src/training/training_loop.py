import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data import DataBatch
from src.models.ddpm import DDPM


# Basic ML training loop
class Trainer:
    def __init__(self, model: DDPM, lr: float = 1e-3, checkpoints_path=None, log_path=None):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.checkpoints_path = checkpoints_path
        self.log_path = log_path  # 可配置的日志路径


    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 10,
            device: torch.device = torch.device("cpu"),
    ):
        self.model.to(device)
        self.train_losses = []
        self.val_losses_list = []

        # 外层 Epoch 进度条
        epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch", ncols=100)
        
        for epoch in epoch_pbar:
            # 训练一个 Epoch
            train_loss = self.train_epoch(train_loader, epoch, epochs, device)
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate_epoch(val_loader, device)
            self.val_losses_list.append(val_loss)
            self.val_losses = val_loss # 保持与原有属性名兼容

            # 更新外层进度条状态
            epoch_pbar.set_postfix({
                "T-Loss": f"{train_loss:.4f}",
                "V-Loss": f"{val_loss:.4f}"
            })

            # 保存 Checkpoint
            if self.checkpoints_path is not None:
                self.save_checkpoint(epoch)

            # 保存训练日志到 CSV
            if self.log_path is not None:
                metrics = [epoch, train_loss, val_loss]
                data = pd.DataFrame([metrics])
                data.to_csv(self.log_path, mode='a', header=False, index=False)

    def train_epoch(
            self,
            train_loader: DataLoader,
            epoch: int,
            total_epochs: int,
            device: torch.device = torch.device("cpu"),
    ) -> float:
        self.model.train()
        epoch_loss = 0
        
        # 内层 Batch 进度条
        batch_pbar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch+1}/{total_epochs}]", 
            leave=False, 
            ncols=100,
            unit="batch"
        )
        
        for i, batch in enumerate(batch_pbar):
            loss = self.train_batch(batch, device)
            epoch_loss += loss.item()
            
            # 实时更新 Batch 损失
            if i % 5 == 0: # 每 5 个 batch 更新一次显示，避免刷新过快
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return epoch_loss / len(train_loader)


    def train_batch(
            self, batch: DataBatch, device: torch.device = torch.device("cpu")
    ) -> torch.FloatTensor:
        batch = batch.to(device)
        losses = self.model.losses(
            x=batch.x,
            batch=batch.batch,
            h=batch.h,
            context=batch.context,
            edge_index=batch.edge_index,
        )
        loss = losses.mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def validate_epoch(
            self, val_loader: DataLoader, device: torch.device = torch.device("cpu")
    ) -> float:
        self.model.eval()

        all_losses = []
        for batch in val_loader:
            all_losses.append(self.validate_batch(batch, device).data)

        return torch.cat(all_losses).mean().item()

    def validate_batch(
            self, batch: DataBatch, device: torch.device = torch.device("cpu")
    ) -> torch.FloatTensor:
        batch = batch.to(device)
        losses = self.model.losses(
            x=batch.x,
            batch=batch.batch,
            h=batch.h,
            context=batch.context,
            edge_index=batch.edge_index,
        )
        return losses

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_losses": self.val_losses,
            },
            self.checkpoints_path,
        )

    def load_checkpoint(self, checkpoint_path, device=torch.device("cpu")):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.val_losses = checkpoint["val_losses"]
        epoch = checkpoint["epoch"]
        return epoch
