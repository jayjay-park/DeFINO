# DeFINO Training Module

This directory contains the training framework for the DeFINO (Derivative-based Fisher-score Informed Neural Operator) project. The training module implements Fourier Neural Operators (FNO) for fluid flow simulation in porous media.

## Environment Setup

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate conda environment
conda env create -f ../environment.yml
conda activate defino-env

# Install additional dependencies
pip install -U 'neptune>=1.0'
pip install lightning
pip install omegaconf
```

## Data Preparation

### Option 1: Download Existing Dataset

If you have access to the Georgia Tech server:

```bash
# Copy data from GT server to local machine
scp -r username@cruyff.cc.gatech.edu:/net/slimdata/jayjaydata/num_ev_8 ./
scp -r username@cruyff.cc.gatech.edu:/net/slimdata/jayjaydata/num_ev_8_stateonly ./
scp username@cruyff.cc.gatech.edu:/net/slimdata/jayjaydata/rescaled_200_fields.h5 ./
```

### Option 2: Transfer Data to Cloud Instance

```bash
# Copy data from local machine to cloud instance
scp -i /path/to/your-key.pem -r ./num_ev_8 username@your-instance:/home/username/DeFINO/data_generation/src/
scp -i /path/to/your-key.pem -r ./num_ev_8_stateonly username@your-instance:/home/username/DeFINO/data_generation/src/
scp -i /path/to/your-key.pem ./rescaled_200_fields.h5 username@your-instance:/home/username/DeFINO/data_generation/src/
```

For incremental synchronization:

```bash
rsync -avz --ignore-existing -e "ssh -i /path/to/your-key.pem" \
  ./num_ev_8/states_sample/* \
  username@your-instance:/home/username/DeFINO/data_generation/src/num_ev_8
```

## Configuration

Set the required environment variables:

```bash
# Set Neptune API token for experiment tracking
export NEPTUNE_API_TOKEN="your-neptune-api-token"

# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Training

Run the training script with a configuration file:

```bash
python main.py --config configs/your-config.yaml
```

## Available Models and Configurations

The training module includes:
- `model.py`: Implementation of the GCS (Gravity-Capillary-System) FNO model
- `dataset.py`: Data loading and preprocessing utilities
- `callbacks.py`: Custom visualization callbacks for training monitoring
- `configs/`: Directory containing example configuration files

## Visualization and Monitoring

Training progress can be monitored using:
- Neptune.ai dashboard (if configured)
- Generated visualization plots in the output directory
- Training logs in the logs directory

## Output Structure

The training creates the following directories:
- `checkpoints/`: Saved model checkpoints
- `logs/`: Training logs
- `output/`: Visualization outputs and metrics
