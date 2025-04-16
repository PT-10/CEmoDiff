import torch


def pretrain_kmeans(model, train_loader, num_samples=100000):
    """Pretrain K-means for semantic encoder"""
    print("Pretraining K-means for semantic encoder...")
    model.eval()
    
    features = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Collecting HuBERT features"):
            waveform = batch['waveform'].to(device)
            
            # Extract HuBERT features
            hubert_features = model.semantic_encoder.hubert(waveform.squeeze(1)).last_hidden_state
            
            # Flatten and add to collection
            flat_features = hubert_features.reshape(-1, hubert_features.size(-1)).cpu().numpy()
            features.append(flat_features)
            
            sample_count += flat_features.shape[0]
            if sample_count >= num_samples:
                break
    
    # Concatenate all features
    all_features = np.vstack(features)
    print(f"Collected {all_features.shape[0]} HuBERT feature vectors")
    
    # Subsample if too many
    if all_features.shape[0] > num_samples:
        indices = np.random.choice(all_features.shape[0], num_samples, replace=False)
        all_features = all_features[indices]
    
    # Train K-means
    print("Training K-means...")
    model.semantic_encoder.train_kmeans(all_features)
    print("K-means training completed")


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    speaker_loss_sum = 0
    contrastive_loss_sum = 0
    semantic_loss_sum = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # Move data to device
        mel_spec = batch['mel_spectrogram'].to(device)
        waveform = batch['waveform'].to(device)
        speaker_ids = batch['speaker_id'].to(device)
        
        # Forward pass
        global_speaker_emb, semantic_repr = model(mel_spec, waveform)
        
        # Get timbre tokens
        _, timbre_tokens = model.timbre_encoder(mel_spec)
        
        # Calculate loss
        loss_dict = criterion(global_speaker_emb, semantic_repr, timbre_tokens, speaker_ids)
        loss = loss_dict['total_loss']
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        torch.nn.utils.clip_grad_norm_(criterion.get_parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        speaker_loss_sum += loss_dict['speaker_loss']
        contrastive_loss_sum += loss_dict['contrastive_loss']
        semantic_loss_sum += loss_dict['semantic_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'spk_loss': loss_dict['speaker_loss'],
            'cont_loss': loss_dict['contrastive_loss'],
            'sem_loss': loss_dict['semantic_loss']
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    avg_speaker_loss = speaker_loss_sum / len(train_loader)
    avg_contrastive_loss = contrastive_loss_sum / len(train_loader)
    avg_semantic_loss = semantic_loss_sum / len(train_loader)
    
    # Log metrics
    if config['wandb']:
        wandb.log({
            'train_loss': avg_loss,
            'train_speaker_loss': avg_speaker_loss,
            'train_contrastive_loss': avg_contrastive_loss,
            'train_semantic_loss': avg_semantic_loss,
            'epoch': epoch
        })
    
    return avg_loss


def validate(model, val_loader, criterion, epoch):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    speaker_loss_sum = 0
    contrastive_loss_sum = 0
    semantic_loss_sum = 0
    
    # For speaker classification accuracy
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for batch in progress_bar:
            # Move data to device
            mel_spec = batch['mel_spectrogram'].to(device)
            waveform = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # Forward pass
            global_speaker_emb, semantic_repr = model(mel_spec, waveform)
            
            # Get timbre tokens
            _, timbre_tokens = model.timbre_encoder(mel_spec)
            
            # Calculate loss
            loss_dict = criterion(global_speaker_emb, semantic_repr, timbre_tokens, speaker_ids)
            loss = loss_dict['total_loss']
            
            # Update metrics
            total_loss += loss.item()
            speaker_loss_sum += loss_dict['speaker_loss']
            contrastive_loss_sum += loss_dict['contrastive_loss']
            semantic_loss_sum += loss_dict['semantic_loss']
            
            # Get speaker predictions for accuracy calculation
            speaker_logits = criterion.speaker_classifier(global_speaker_emb)
            _, preds = torch.max(speaker_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(speaker_ids.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': loss.item()
            })
    
    # Calculate average losses
    avg_loss = total_loss / len(val_loader)
    avg_speaker_loss = speaker_loss_sum / len(val_loader)
    avg_contrastive_loss = contrastive_loss_sum / len(val_loader)
    avg_semantic_loss = semantic_loss_sum / len(val_loader)
    
    # Calculate speaker classification accuracy
    speaker_accuracy = accuracy_score(all_labels, all_preds)
    
    # Log metrics
    if config['wandb']:
        wandb.log({
            'val_loss': avg_loss,
            'val_speaker_loss': avg_speaker_loss,
            'val_contrastive_loss': avg_contrastive_loss,
            'val_semantic_loss': avg_semantic_loss,
            'val_speaker_accuracy': speaker_accuracy,
            'epoch': epoch
        })
    
    print(f"Validation: Loss={avg_loss:.4f}, Speaker Acc={speaker_accuracy:.4f}")
    
    return avg_loss, speaker_accuracy


def save_checkpoint(model, criterion, optimizer, scheduler, epoch, val_loss, val_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'criterion_speaker_state_dict': criterion.speaker_classifier.state_dict(),
        'criterion_semantic_state_dict': criterion.semantic_predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, criterion, optimizer, scheduler, filepath):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.speaker_classifier.load_state_dict(checkpoint['criterion_speaker_state_dict'])
    criterion.semantic_predictor.load_state_dict(checkpoint['criterion_semantic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    best_val_acc = checkpoint.get('val_acc', 0.0)
    
    return start_epoch, best_val_loss, best_val_acc