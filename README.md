# CEmoDiff
Speech Understanding Course Project

CEmoDiff is a project focused on speech understanding, featuring models for pitch quantization, timbre encoding, and semantic representation for controllable emotion generation in speech. The project includes training scripts, pre-processing utilities, and model implementations.

---

## Directory Structure
```
CEmoDiff/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── checkpoints/
│   │   ├── pitch_vqvae_checkpoints/
│   │   ├── precomputed_f0.pkl
│   │   └── speaker_mean_log_f0.pkl
│   ├── config/
│   │   ├── pitch_vqvae_training_config.yaml
│   │   └── semantic_timbre_training_config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pitch_vqvae_dataloader.py
│   │   └── semantic_timbre_dataloader.py
│   ├── dataset/
│   │   └── LibriTTS/train-clean-100
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pitch_vqvae.py
│   │   ├── semantic_encoder.py
│   │   ├── speech_model_semi.py
│   │   └── timbre_encoder.py
│   ├── notebooks/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_pitch_vqvae.py
│   │   └── train_semantic_timbre.py
│   └── utils/
│       ├── __init__.py
│       ├── loss_functions.py
│       ├── mlp.py
│       └── utilities.py
```

## Setup Instructions

### Prerequisites

Ensure you have:
- Python 3.9

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/CEmoDiff.git
   cd CEmoDiff
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   Ensure the LibriTTS dataset is downloaded and placed in the directory `src/dataset/LibriTTS/train-clean-100`.

4. Verify the configuration files:
   - Training configurations for `pitch-vqvae` are in `src/config/pitch_vqvae_training_config.yaml`.
   - Training configurations for `semantic-timbre` are in `src/config/semantic_timbre_training_config.yaml`.

---

## Training

### Train the Pitch-VQVAE Model

To train the `pitch-vqvae` model, run the following command:
```bash
python -m src.training.train_pitch_vqvae
```

### Train the Semantic-Timbre Model

To train the `semantic-timbre` model, run the following command:
```bash
python -m src.training.train_semantic_timbre
```

## Preprocessing

### Precompute F0 and Speaker Mean Log F0

The `AudioDataset` class in `src/data/pitch_vqvae_dataloader.py` supports precomputing F0 values and speaker mean log F0. These are saved as `precomputed_f0.pkl` and `speaker_mean_log_f0.pkl`.

---