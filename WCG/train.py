import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from .model import GatingModuleDeepFDConv
from .dataset import WeatherDataset, get_transforms

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_predictions/total_samples):.4f}")
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_predictions/total_samples):.4f}")
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_path = os.path.join(config['experiment_dir'], config['exp_name'])
    os.makedirs(exp_path, exist_ok=True)
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_path, 'tensorboard'))
  
    train_transform = get_transforms(img_size=config['img_size'], is_train=True)
    val_transform = get_transforms(img_size=config['img_size'], is_train=False)

    full_dataset_base = WeatherDataset(csv_file=config['csv_file'], img_dir=config['img_dir'], transform=None)
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset_base, [0.9, 0.1], generator=generator)

    class DatasetWrapper(Dataset):
        def __init__(self, subset, transform): self.subset, self.transform = subset, transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            return self.transform(image), label
            
    train_dataset = DatasetWrapper(train_subset, train_transform)
    val_dataset = DatasetWrapper(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=False)
    
    model = GatingModuleDeepFDConv(
        model_name=config['backbone_name'],
        num_classes=config['num_classes']
    ).to(device)
    
    if config.get('pretrained_weights_path'):
        weights_path = config['pretrained_weights_path']
        if os.path.exists(weights_path):
            pretrained_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in pretrained_dict: 
                pretrained_dict = pretrained_dict['model']
            
            missing_keys, unexpected_keys = model.backbone.load_state_dict(pretrained_dict, strict=False)
            if missing_keys:
                print(f"  - 缺失的键 (应为FC层): {missing_keys}")
            if unexpected_keys:
                print(f"  - 意外的键 (应为空): {unexpected_keys}")
        else:
            print(f"警告: 找不到 backbone 权重 '{weights_path}'。")

    backbone_params = model.backbone.parameters()
    new_modules_params = list(model.attention.parameters()) + list(model.head.parameters())
    
    lr = config['learning_rate']
    backbone_lr = config.get('backbone_lr', lr / 10.0)
    param_groups = [{'params': backbone_params, 'lr': backbone_lr}, {'params': new_modules_params, 'lr': lr}]
    print(f"  FDConv, Head: {lr}")
    print(f"  Backbone: {backbone_lr}")
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=5e-4)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    warmup_epochs = 5
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=1e-6) if config['epochs'] > warmup_epochs else None

    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        if epoch < warmup_epochs:
            target_lr_groups = [backbone_lr, lr]
            initial_lr_groups = [config.get('initial_lr_for_warmup', 1e-7)] * 2
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = initial_lr_groups[i] + (target_lr_groups[i] - initial_lr_groups[i]) * (epoch + 1) / warmup_epochs
        elif main_scheduler:
            main_scheduler.step()
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LRs: {current_lrs}")
        
        writer.add_scalar('Learning_Rate/backbone', current_lrs[0], epoch)
        writer.add_scalar('Learning_Rate/new_heads', current_lrs[1], epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(checkpoint_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"发现更高验证准确率 ({best_val_acc:.4f})，模型已保存到: {save_path}")

    final_save_path = os.path.join(checkpoint_path, 'latest_model.pth')
    torch.save(model.state_dict(), final_save_path)
    writer.close()
    print(f"Finished: {final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepFDConv")
    parser.add_argument('--config', type=str, required=True, help='path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    exp_name_base = config.get('exp_name_base', 'Final_Test_DeepFDConv')
    config['exp_name'] = f"{exp_name_base}_{time.strftime('%Y%m%d-%H%M%S')}"

    main(config)
