import os
import glob
import torch
import random
import warnings
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


class SpeechDataset(Dataset):
    """
    Dataset for loading speech audio files for the combined Timbre and Semantic Encoders.
    Handles preprocessing of waveforms and mel-spectrograms.
    """
    def __init__(self, 
                 root_dir,
                 metadata_file=None,
                 sample_rate=16000,
                 n_mels=80,
                 mel_fmin=0,
                 mel_fmax=8000,
                 hop_length=256,
                 win_length=1024,
                 n_fft=1024,
                 segment_length=16000,  # 1 second at 16kHz
                 augment=False,
                 speaker_dict_file=None):
        """
        Args:
            root_dir: Directory with all the audio files
            metadata_file: Path to CSV file with metadata (speaker_id, filepath)
            sample_rate: Target sampling rate
            n_mels: Number of mel filterbanks
            mel_fmin: Minimum frequency for mel filterbank
            mel_fmax: Maximum frequency for mel filterbank
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_fft: FFT size
            segment_length: Length of audio segments in samples
            augment: Whether to apply data augmentation
            speaker_dict_file: Path to JSON file with speaker ID mapping
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.augment = augment
        
        # Mel spectrogram parameters
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        
        # Load metadata
        if metadata_file is not None:
            self.metadata = pd.read_csv(metadata_file)
        else:
            # If no metadata file, scan directory for audio files
            self._scan_directory()
        
        # Create speaker dictionary
        if speaker_dict_file is not None and os.path.exists(speaker_dict_file):
            import json
            with open(speaker_dict_file, 'r') as f:
                self.speaker_dict = json.load(f)
        else:
            self._create_speaker_dict()
            
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mels,
            center=True,
            power=1.0,  # amplitude spectrogram, not power
        )
        
        # Amplitude to DB conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        

    def _scan_directory(self):
        """Scan LibriTTS directory for audio files and create metadata."""
        print("Scanning directory for audio files...")
        audio_files = []
        speaker_ids = []
    
        # List speaker directories (limit to first 100)
        speaker_dirs = sorted([
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])[:100]  # Limit to first 100 speaker directories

        # print("Speaker dir", speaker_dirs)
        # Gather all audio file paths
        audio_paths = []
        for speaker_dir in speaker_dirs:
            for ext in ['wav', 'flac', 'mp3']:
                audio_paths.extend(glob.glob(os.path.join(speaker_dir, f'**/*.{ext}'), recursive=True))
    
        print(f"Found {len(audio_paths)} audio files in first 100 speaker directories.")
    
        # Extract speaker IDs and construct metadata
        for path in audio_paths:
            speaker_id = os.path.basename(path).split('_')[0]
            audio_files.append(path)
            speaker_ids.append(speaker_id)
    
        # Store metadata as a DataFrame
        self.metadata = pd.DataFrame({
            'speaker_id': speaker_ids,
            'filepath': audio_files
        })
        
        print(f"Created metadata with {len(self.metadata)} entries from {len(set(speaker_ids))} speakers.")

    
    def _create_speaker_dict(self):
        """Create dictionary mapping speaker IDs to integers"""
        unique_speakers = self.metadata['speaker_id'].unique()
        self.speaker_dict = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        
        # Save speaker dictionary for future use
        import json
        with open( 'speaker_dict.json', 'w') as f:
            json.dump(self.speaker_dict, f)
            
        print(f"Created speaker dictionary with {len(self.speaker_dict)} speakers")
    
    def __len__(self):
        return len(self.metadata)
    
    def _load_audio(self, audio_path):
        """Load audio file and resample if necessary"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
            return waveform
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a short silence as fallback
            return torch.zeros(1, self.sample_rate)
    
    def _extract_mel_spectrogram(self, waveform):
        """Extract mel spectrogram from waveform"""
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [C, n_mels, T]
        
        # Convert to dB scale
        mel_spec = self.amplitude_to_db(mel_spec)  # [C, n_mels, T]
        
        # Normalize to [-1, 1] range
        mel_spec = (mel_spec + 80) / 80  # normalize considering 80 dB dynamic range
        mel_spec = 2 * mel_spec - 1  # rescale to [-1, 1]
        
        return mel_spec
    
    def _random_segment(self, waveform):
        """Randomly segment waveform to fixed length"""
        if waveform.shape[1] <= self.segment_length:
            # Pad if shorter than segment length
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
            return waveform
        else:
            # Random starting point
            start = random.randint(0, waveform.shape[1] - self.segment_length)
            return waveform[:, start:start + self.segment_length]
    
    def _apply_augmentation(self, waveform):
        """Apply simple audio augmentations"""
        # Volume adjustment
        volume_factor = random.uniform(0.8, 1.2)
        waveform = waveform * volume_factor
        
        # Add small noise
        noise_level = random.uniform(0.0, 0.005)
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
        
        # Time stretching using librosa (optional - more computationally expensive)
        # if random.random() < 0.3:
        #     stretch_factor = random.uniform(0.9, 1.1)
        #     waveform_np = waveform.numpy().squeeze()
        #     waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=stretch_factor)
        #     waveform = torch.from_numpy(waveform_stretched).unsqueeze(0)
        
        return waveform
    
    def __getitem__(self, idx):
        """Get a training item"""
        item = self.metadata.iloc[idx]
        audio_path = item['filepath']
        speaker_id = item['speaker_id']
        
        # Load audio
        waveform = self._load_audio(audio_path)
        
        # Apply segmentation
        waveform = self._random_segment(waveform)
        
        # Apply augmentation if enabled
        if self.augment:
            waveform = self._apply_augmentation(waveform)
        
        # Extract mel spectrogram
        mel_spec = self._extract_mel_spectrogram(waveform)
        
        # Get speaker ID as integer
        speaker_idx = self.speaker_dict[speaker_id]
        
        return {
            'waveform': waveform,  # [1, T]
            'mel_spectrogram': mel_spec,  # [1, n_mels, T]
            'speaker_id': speaker_idx,
            'speaker_name': speaker_id,
            'audio_path': audio_path
        }


class SpeechCollator:
    """
    Collator for batching speech data
    """
    def __init__(self, pad_mel=True):
        self.pad_mel = pad_mel
    
    def __call__(self, batch):
        """
        Args:
            batch: List of items from dataset
        
        Returns:
            Batched tensors
        """
        # Extract items
        waveforms = [item['waveform'] for item in batch]
        mel_specs = [item['mel_spectrogram'] for item in batch]
        speaker_ids = torch.tensor([item['speaker_id'] for item in batch], dtype=torch.long)
        speaker_names = [item['speaker_name'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]
        
        # Stack waveforms (they should all be the same length)
        waveforms = torch.stack(waveforms)
        
        # Handle mel spectrograms
        if self.pad_mel:
            # Get max length
            max_len = max(spec.shape[2] for spec in mel_specs)
            
            # Pad to max length
            padded_mel_specs = []
            for spec in mel_specs:
                padding = max_len - spec.shape[2]
                if padding > 0:
                    padded_spec = F.pad(spec, (0, padding))
                else:
                    padded_spec = spec
                padded_mel_specs.append(padded_spec)
            
            # Stack
            mel_specs = torch.stack(padded_mel_specs)
        else:
            # Just stack (must be same length)
            mel_specs = torch.stack(mel_specs)
        
        return {
            'waveform': waveforms,  # [B, 1, T]
            'mel_spectrogram': mel_specs,  # [B, 1, n_mels, T]
            'speaker_id': speaker_ids,  # [B]
            'speaker_name': speaker_names,  # List[str]
            'audio_path': audio_paths  # List[str]
        }


def create_dataloaders(root_dir, 
                       batch_size=16,
                       num_workers=4,
                       metadata_file=None,
                       val_split=0.1,
                       test_split=0.1,
                       seed=42,
                       **dataset_kwargs):
    """
    Create training and validation dataloaders
    
    Args:
        root_dir: Directory with audio files
        batch_size: Batch size
        num_workers: Number of workers for data loading
        metadata_file: Path to metadata CSV file
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
        **dataset_kwargs: Additional arguments for SpeechDataset
    
    Returns:
        train_loader, val_loader, test_loader: DataLoaders
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create dataset
    dataset = SpeechDataset(root_dir, metadata_file=metadata_file, **dataset_kwargs)
    
    # Split dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    test_size = int(np.floor(test_split * dataset_size))
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - val_size - test_size
    
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create samplers
    from torch.utils.data import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create collator
    collator = SpeechCollator()
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Created dataloaders: {len(train_loader)} training batches, "
          f"{len(val_loader)} validation batches, {len(test_loader)} test batches")
    
    return train_loader, val_loader, test_loader