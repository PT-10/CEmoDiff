import torch
from torch.utils.data import Dataset

import os
import numba
import librosa
import numpy as np


# Helper function for parallel processing of F0 extraction
# Helper function for processing voiced F0
@numba.jit(nopython=True)
def process_voiced_f0(f0, voiced_flag):
    """
    Process voiced F0 values efficiently with Numba
    """
    voiced_f0 = f0[voiced_flag]
    log_f0_values = []
    
    for val in voiced_f0:
        if val > 0:
            log_f0_values.append(np.log(val))
    
    return log_f0_values


def extract_f0_for_file(path, sr, min_f0, max_f0, hop_length):
    """
    Extract F0 for a single file - sequential version
    """
    try:
        # Load audio
        y, _ = librosa.load(path, sr=sr)
        
        # Extract F0
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=min_f0,
            fmax=max_f0,
            hop_length=hop_length,
            sr=sr
        )
        
        # Get speaker ID
        speaker_id = os.path.basename(path).split('_')[0]
        
        # Calculate log F0 for voiced frames
        log_f0_values = []
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 0:
            log_f0 = np.log(voiced_f0[voiced_f0 > 0])
            log_f0_values = log_f0.tolist()
        
        return speaker_id, log_f0_values
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, []

import pickle



# Pre-load and cache audio data to reduce disk I/O
class AudioCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, path, sr):
        if path in self.cache:
            self.access_count[path] += 1
            return self.cache[path]
        
        # Load audio
        y, _ = librosa.load(path, sr=sr)
        
        # Manage cache size
        if len(self.cache) >= self.max_size:
            # Remove least recently accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        # Cache new item
        self.cache[path] = y
        self.access_count[path] = 1
        
        return y


class AudioDataset(Dataset):
    """
    Dataset for loading audio files and extracting F0 with optimizations
    but without multiprocessing
    """
    def __init__(self, audio_paths, hop_length=256, sr=22050, min_f0=50, max_f0=600, segment_length=8192, 
                 cache_size=50, precompute=True, load_precomputed=False, f0_cache_path="src/checkpoints/precomputed_f0.pkl", 
             mean_f0_path="src/checkpoints/speaker_mean_log_f0.pkl"):
        self.audio_paths = audio_paths
        self.hop_length = hop_length
        self.sr = sr
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.segment_length = segment_length
        
        print(f"Initializing AudioDataset with {len(audio_paths)} files")
        
        # Audio cache to reduce disk I/O
        self.audio_cache = AudioCache(max_size=cache_size)
        
        # Get speaker IDs from filenames (assuming format speakerID_*.wav)
        self.speaker_ids = {}
        for i, path in enumerate(self.audio_paths):
            speaker_id = os.path.basename(path).split('_')[0]
            if speaker_id not in self.speaker_ids:
                self.speaker_ids[speaker_id] = len(self.speaker_ids)
        
        print(f"Found {len(self.speaker_ids)} unique speakers")

        # Optionally pre-extract F0 for all files
        self.precomputed_f0 = {}
        # Precompute speaker mean log F0 using sequential processing
        self.speaker_mean_log_f0 = {}
        if load_precomputed:
            self.load_precomputes(f0_path=f0_cache_path, mean_path=mean_f0_path)  
        elif precompute:
            print("-" * 50)
            print("STAGE 1: Computing speaker mean log F0...")
            print("-" * 50)
            self._compute_speaker_mean_log_f0_sequential()
            print("COMPLETED: Speaker mean log F0 computation")
            print("-" * 50)
            print("-" * 50)
            print("STAGE 2: Pre-extracting F0 for all files...")
            print("-" * 50)
            self._precompute_f0_sequential()
            print("COMPLETED: F0 pre-extraction")
            print("-" * 50)
            self.save_precomputes(f0_cache_path, mean_f0_path)
            
        print("AudioDataset initialization complete!")


    

    def save_precomputes(self, f0_path="src/checkpoints/precomputed_f0.pkl", mean_path="src/checkpoints/speaker_mean_log_f0.pkl"):
        with open(f0_path, "wb") as f:
            pickle.dump(self.precomputed_f0, f)
        with open(mean_path, "wb") as f:
            pickle.dump(self.speaker_mean_log_f0, f)
        print(f"Saved precomputed F0 to {f0_path}")
        print(f"Saved speaker mean log F0 to {mean_path}")

    def load_precomputes(self, f0_path="src/checkpoints/precomputed_f0.pkl", mean_path="src/checkpoints/speaker_mean_log_f0.pkl"):
        try:
            with open(f0_path, "rb") as f:
                self.precomputed_f0 = pickle.load(f)
            with open(mean_path, "rb") as f:
                self.speaker_mean_log_f0 = pickle.load(f)
            print(f"Loaded precomputed F0 from {f0_path}")
            print(f"Loaded speaker mean log F0 from {mean_path}")
        except Exception as e:
            print(f"Failed to load precomputes: {e}")


    def _compute_speaker_mean_log_f0_sequential(self):
        """
        Precompute mean log F0 for each speaker using sequential processing
        """
        speaker_f0_values = {}
        
        print(f"Computing speaker mean log F0 sequentially...")
        print(f"Processing {len(self.audio_paths)} audio files...")
        
        # Process files with progress bar
        from tqdm import tqdm
        
        for i, path in enumerate(tqdm(self.audio_paths, desc="Computing speaker mean log F0")):
            speaker_id, log_f0_values = extract_f0_for_file(
                path,
                sr=self.sr,
                min_f0=self.min_f0,
                max_f0=self.max_f0,
                hop_length=self.hop_length
            )
            
            # Store results
            if speaker_id is not None:
                if speaker_id not in speaker_f0_values:
                    speaker_f0_values[speaker_id] = []
                speaker_f0_values[speaker_id].extend(log_f0_values)
            
            # # Print progress occasionally
            # if (i+1) % 20 == 0 or i == len(self.audio_paths) - 1:
            #     print(f"Processed {i+1}/{len(self.audio_paths)} files ({(i+1)/len(self.audio_paths)*100:.1f}%)")
        
        print("F0 extraction complete. Computing speaker means...")
        
        # Compute mean log F0 for each speaker
        for speaker_id, values in speaker_f0_values.items():
            if values:
                self.speaker_mean_log_f0[speaker_id] = np.mean(values)
                print(f"Speaker {speaker_id}: Mean log F0 = {self.speaker_mean_log_f0[speaker_id]:.4f} ({len(values)} frames)")
            else:
                self.speaker_mean_log_f0[speaker_id] = 0.0
                print(f"Speaker {speaker_id}: No valid F0 frames found")
    
    def _precompute_f0_sequential(self):
        """
        Pre-extract F0 for all files sequentially
        """
        print(f"Pre-extracting F0 sequentially...")
        
        # Process files sequentially with tqdm
        from tqdm import tqdm
        
        successful = 0
        failed = 0
        
        for i, path in enumerate(tqdm(self.audio_paths, desc="Pre-extracting F0")):
            try:
                # Load audio
                y, _ = librosa.load(path, sr=self.sr)
                
                # Extract F0
                f0, voiced_flag, _ = librosa.pyin(
                    y,
                    fmin=self.min_f0,
                    fmax=self.max_f0,
                    hop_length=self.hop_length,
                    sr=self.sr
                )
                
                # Store result
                self.precomputed_f0[path] = (f0, voiced_flag)
                successful += 1
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                failed += 1
            
            # # Print progress occasionally
            # if (i+1) % 20 == 0 or i == len(self.audio_paths) - 1:
            #     print(f"Processed {i+1}/{len(self.audio_paths)} files ({(i+1)/len(self.audio_paths)*100:.1f}%)")
        
        print(f"Successfully pre-computed F0 for {successful} files, failed for {failed} files")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        speaker_id = os.path.basename(path).split('_')[0]
        
        # Try to use precomputed F0 if available
        if path in self.precomputed_f0:
            f0, voiced_flag = self.precomputed_f0[path]
        else:
            # Load audio from cache if possible
            y = self.audio_cache.get(path, self.sr)
            
            # Random segment for training
            if len(y) > self.segment_length:
                start = np.random.randint(0, len(y) - self.segment_length)
                y = y[start:start + self.segment_length]
            else:
                # Pad if too short
                y = np.pad(y, (0, max(0, self.segment_length - len(y))))
            
            # Extract F0
            f0, voiced_flag, _ = librosa.pyin(
                y, 
                fmin=self.min_f0,
                fmax=self.max_f0,
                hop_length=self.hop_length,
                sr=self.sr
            )
        expected_len = self.segment_length // self.hop_length + 1

        # Pad or trim f0 and voiced_flag
        def pad_or_trim(x, target_len):
            if len(x) < target_len:
                return np.pad(x, (0, target_len - len(x)))
            else:
                return x[:target_len]
    
        f0 = pad_or_trim(f0, expected_len)
        voiced_flag = pad_or_trim(voiced_flag.astype(float), expected_len).astype(bool)
        
        # Convert to tensor
        f0_tensor = torch.zeros(len(f0), dtype=torch.float32)
        f0_tensor[voiced_flag] = torch.tensor(f0[voiced_flag], dtype=torch.float32)
        
        # Create voiced mask
        voiced_mask = torch.tensor(voiced_flag.astype(float), dtype=torch.float32)
        
        # Compute SMN log F0
        smn_log_f0 = torch.zeros_like(f0_tensor)
        log_f0 = torch.zeros_like(f0_tensor)
        
        # Take log of voiced frames
        log_f0[voiced_mask > 0] = torch.log(f0_tensor[voiced_mask > 0])
        
        # Normalize by speaker mean
        speaker_mean = self.speaker_mean_log_f0[speaker_id]
        smn_log_f0[voiced_mask > 0] = log_f0[voiced_mask > 0] - speaker_mean
        
        return {
            'smn_log_f0': smn_log_f0,
            'voiced_mask': voiced_mask,
            'speaker_id': self.speaker_ids[speaker_id]
        }
