import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json
import multiprocessing
from src.data.pitch_vqvae_dataloader import AudioDataset
from src.models.pitch_vqvae import PitchVQVAE


def train(model, train_loader, optimizer, device, epoch, log_interval=10):
    """
    Training loop for one epoch (without mixed precision)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_commitment_loss = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, batch in pbar:
        # Move data to device
        smn_log_f0 = batch['smn_log_f0'].to(device)
        voiced_mask = batch['voiced_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        _, _, loss, loss_dict = model(smn_log_f0, voiced_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Extract loss components
        recon_loss = loss_dict['recon_loss']
        vq_loss = loss_dict['vq_loss']
        commitment_loss = loss_dict['commitment_loss']
        total_batch_loss = loss_dict['total_loss']

        # Ensure values are scalars
        total_loss += total_batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_commitment_loss += commitment_loss.item()

        # Logging
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'recon_loss': total_recon_loss / (batch_idx + 1),
                'vq_loss': total_vq_loss / (batch_idx + 1),
                'commit_loss': total_commitment_loss / (batch_idx + 1)
            })

    return {
        'loss': total_loss / len(train_loader),
        'recon_loss': total_recon_loss / len(train_loader),
        'vq_loss': total_vq_loss / len(train_loader),
        'commitment_loss': total_commitment_loss / len(train_loader)
    }


def validate(model, val_loader, device):
    """
    Validation loop
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_commitment_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            smn_log_f0 = batch['smn_log_f0'].to(device)
            voiced_mask = batch['voiced_mask'].to(device)
            
            # Forward pass
            _, _, _, loss_dict = model(smn_log_f0, voiced_mask)
            
            # Update statistics
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
            total_vq_loss += loss_dict['vq_loss']
            total_commitment_loss += loss_dict['commitment_loss']
    
    return {
        'loss': total_loss / len(val_loader),
        'recon_loss': total_recon_loss / len(val_loader),
        'vq_loss': total_vq_loss / len(val_loader),
        'commitment_loss': total_commitment_loss / len(val_loader)
    }


def visualize_reconstructions(model, val_loader, device, num_samples=5, save_path=None):
    """
    Visualize original and reconstructed pitch contours
    """
    model.eval()
    
    # Create figure only once
    plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            # Move data to device
            smn_log_f0 = batch['smn_log_f0'].to(device)
            voiced_mask = batch['voiced_mask'].to(device)
            
            # Forward pass
            x_recon, indices, _, _ = model(smn_log_f0, voiced_mask)
            
            # Move back to CPU for plotting
            smn_log_f0 = smn_log_f0[0].cpu().numpy()
            x_recon = x_recon[0].cpu().numpy()
            voiced_mask = voiced_mask[0].cpu().numpy()
            indices = indices[0].cpu().numpy()
            
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.plot(smn_log_f0 * voiced_mask, label='Original')
            plt.title(f'Original SMN log F0 - Sample {i+1}')
            plt.grid(True)
            
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.plot(x_recon * voiced_mask, label='Reconstructed')
            plt.title(f'Reconstructed SMN log F0 - Sample {i+1}')
            plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_codebook_usage(model, val_loader, device, save_path=None):
    """
    Visualize codebook usage distribution
    """
    model.eval()
    
    indices_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting codebook usage"):
            # Move data to device
            smn_log_f0 = batch['smn_log_f0'].to(device)
            voiced_mask = batch['voiced_mask'].to(device)
            
            # Forward pass
            _, indices, _, _ = model(smn_log_f0, voiced_mask)
            
            # Collect indices for voiced frames
            for i in range(indices.size(0)):
                voiced_indices = indices[i][voiced_mask[i] > 0].cpu().numpy()
                indices_list.extend(voiced_indices)
    
    # Count usage frequency
    unique_indices, counts = np.unique(indices_list, return_counts=True)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(unique_indices, counts)
    plt.xlabel('Codebook Index')
    plt.ylabel('Usage Count')
    plt.title('Codebook Usage Distribution')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """
    Save model checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set number of threads for CPU operations
    # torch.set_num_threads(min(config.get('num_threads', 4), multiprocessing.cpu_count()))
    # print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    

    audio_paths = []
    speaker_dirs = sorted([
        os.path.join(config['data_dir'], d)
        for d in os.listdir(config['data_dir'])
        if os.path.isdir(os.path.join(config['data_dir'], d))
    ])[:100]  # Take only the first 100 speaker directories
    
    for speaker_dir in speaker_dirs:
        for ext in ['wav', 'flac', 'mp3']:
            audio_paths.extend(glob.glob(os.path.join(speaker_dir, f'**/*.{ext}'), recursive=True))

    
    print(f"Found {len(audio_paths)} audio files")
    
    # Split into train and validation sets
    np.random.seed(config['seed'])
    np.random.shuffle(audio_paths)
    
    split_idx = int(len(audio_paths) * 0.9)
    train_paths = audio_paths[:split_idx]
    val_paths = audio_paths[split_idx:]
    
    print(f"Training set: {len(train_paths)} files")
    print(f"Validation set: {len(val_paths)} files")
    
    # Create datasets and data loaders with optimizations
    train_dataset = AudioDataset(
        train_paths,
        hop_length=config['hop_length'],
        sr=config['sample_rate'],
        min_f0=config['min_f0'],
        max_f0=config['max_f0'],
        segment_length=config['segment_length'],
        cache_size=config.get('cache_size', 50),
        precompute=config.get('precompute_f0', True)
    )
    
    val_dataset = AudioDataset(
        val_paths,
        hop_length=config['hop_length'],
        sr=config['sample_rate'],
        min_f0=config['min_f0'],
        max_f0=config['max_f0'],
        segment_length=config['segment_length'],
        cache_size=config.get('cache_size', 50),
        precompute=config.get('precompute_f0', True)
    )
    
    # Use efficient data loading with pinned memory and optimal worker count
    # Number of workers should be balanced based on CPU cores and memory
    optimal_workers = min(config.get('num_workers', 4), multiprocessing.cpu_count())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=2 if optimal_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=2 if optimal_workers > 0 else None
    )
    
    # Create model
    model = PitchVQVAE(
        codebook_size=config['codebook_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        reconstruction_coef=config['reconstruction_coef'],
        commitment_coef=config['commitment_coef'],
        vq_coef=config['vq_coef']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss_dict = train(model, train_loader, optimizer, device, epoch, config['log_interval'])
        train_losses.append(train_loss_dict)
        
        # Validate
        val_loss_dict = validate(model, val_loader, device)
        val_losses.append(val_loss_dict)
        
        # Update learning rate
        scheduler.step(val_loss_dict['loss'])
        
        # Print losses
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"  Train loss: {train_loss_dict['loss']:.6f}")
        print(f"  Validation loss: {val_loss_dict['loss']:.6f}")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss_dict['loss'], config['save_dir'])
        
        # Save best model
        if val_loss_dict['loss'] < best_val_loss:
            best_val_loss = val_loss_dict['loss']
            save_checkpoint(model, optimizer, epoch, val_loss_dict['loss'], os.path.join(config['save_dir'], 'best'))
        
        # Visualize reconstructions and codebook usage
        if epoch % config['viz_interval'] == 0:
            visualize_reconstructions(
                model,
                val_loader,
                device,
                num_samples=5,
                save_path=os.path.join(config['save_dir'], f'reconstructions_epoch_{epoch}.png')
            )
            
            visualize_codebook_usage(
                model,
                val_loader,
                device,
                save_path=os.path.join(config['save_dir'], f'codebook_usage_epoch_{epoch}.png')
            )
    
    # Save final model
    save_checkpoint(model, optimizer, config['epochs'], val_loss_dict['loss'], config['save_dir'])
    
    # Save loss history
    train_losses_df = pd.DataFrame(train_losses)
    val_losses_df = pd.DataFrame(val_losses)
    
    train_losses_df.to_csv(os.path.join(config['save_dir'], 'train_losses.csv'), index=False)
    val_losses_df.to_csv(os.path.join(config['save_dir'], 'val_losses.csv'), index=False)
    
    # Plot loss curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses_df['loss'], label='Train')
    plt.plot(val_losses_df['loss'], label='Validation')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(train_losses_df['recon_loss'], label='Train')
    plt.plot(val_losses_df['recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(train_losses_df['vq_loss'], label='Train')
    plt.plot(val_losses_df['vq_loss'], label='Validation')
    plt.title('VQ Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(train_losses_df['commitment_loss'], label='Train')
    plt.plot(val_losses_df['commitment_loss'], label='Validation')
    plt.title('Commitment Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'loss_curves.png'))
    plt.show()

if __name__ == "__main__":
    from src.utils.utilities import get_config

    config = get_config("src/config/pitch_vqvae_training_config.yaml")
    print("Training with the following configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    main(config)
