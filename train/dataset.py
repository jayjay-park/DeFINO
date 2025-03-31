import os
import torch
import numpy as np
import h5py
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class CustomDataset(torch.utils.data.Dataset):
    """Dataset for GCS saturation prediction with additional attributes for VJP training."""
    
    def __init__(self, x, y, vjp=None, eigvec=None, indices=None):
        """
        Initialize the dataset.
        
        Args:
            x: Input permeability fields
            y: Ground truth saturation fields
            vjp: Vector-Jacobian products (optional)
            eigvec: Eigenvectors (optional)
            indices: Indices of the samples
        """
        self.x = x
        self.y = y
        self.vjp = vjp
        self.eigvec = eigvec
        self.indices = indices

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {
            'x': self.x[idx],
            'y': self.y[idx],
            'idx': self.indices[idx]
        }
        
        if self.vjp is not None:
            item['vjp'] = self.vjp[idx]
            
        if self.eigvec is not None:
            item['eigvec'] = self.eigvec[idx]
            
        return item


class GCSDataLoader:
    """Data loader for GCS dataset that allows creating train and validation splits using offset and limit."""
    
    def __init__(
        self,
        k_file: str,
        states_dir: str,
        batch_size: int = 1,
        nt: int = 5,
        nx: int = 256,
        ny: int = 256,
        num_vec: int = 8,
        num_workers: int = 4,
        vjp_dir: Optional[str] = None,
        eigvec_dir: Optional[str] = None,
        pin_memory: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            k_file: Path to the permeability field file
            states_dir: Directory containing saturation states
            batch_size: Batch size for the dataloaders
            nt: Number of time steps
            nx: Grid size in x-direction
            ny: Grid size in y-direction
            num_vec: Number of vectors for VJP calculation
            num_workers: Number of workers for data loading
            vjp_dir: Directory containing VJP files (optional)
            eigvec_dir: Directory containing eigenvector files (optional)
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.k_file = k_file
        self.states_dir = states_dir
        self.vjp_dir = vjp_dir
        self.eigvec_dir = eigvec_dir
        
        self.batch_size = batch_size
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.num_vec = num_vec
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Verify data sources exist
        self._verify_data_sources()
        
        # Load and preprocess permeability fields
        self.K_transposed, self.set_x = self._load_permeability_fields()
        
        # Determine valid sample indices
        self.sample_indices = self._get_sample_indices()
    
    def _verify_data_sources(self):
        """Check if the data sources exist and are valid."""
        if not os.path.exists(self.k_file):
            raise FileNotFoundError(f"Permeability file not found: {self.k_file}")
        
        if not os.path.exists(self.states_dir):
            raise FileNotFoundError(f"States directory not found: {self.states_dir}")
        
        if self.vjp_dir and not os.path.exists(self.vjp_dir):
            raise FileNotFoundError(f"VJP directory not found: {self.vjp_dir}")
            
        if self.eigvec_dir and not os.path.exists(self.eigvec_dir):
            raise FileNotFoundError(f"Eigenvector directory not found: {self.eigvec_dir}")
    
    def _load_permeability_fields(self):
        """Load and preprocess permeability fields."""
        with h5py.File(self.k_file, "r") as f:
            K = f["K_subset"][:].astype(np.float32)
        
        K_transposed = np.transpose(K, (2, 0, 1))
        K_min = K_transposed.min()
        K_max = K_transposed.max()
        set_x = (K_transposed - K_min) / (K_max - K_min)
        
        return K_transposed, set_x
    
    def _get_sample_indices(self):
        """Determine which samples to include based on exclusion range."""
        sample_indices = []
        exclude_range = (41, 81)
        for s_idx in range(1, len(self.K_transposed) + 1):
            if s_idx >= exclude_range[0] and s_idx <= exclude_range[1]:
                continue
            sample_indices.append(s_idx)
        return sample_indices
    
    def _load_data_subset(self, offset: int, limit: int):
        """
        Load a subset of the data.
        
        Args:
            offset: Starting index for the data subset
            limit: Number of samples to include
            
        Returns:
            Tuple containing the dataset components
        """
        # Ensure we have enough samples
        if len(self.sample_indices) < (offset + limit):
            raise ValueError(f"Not enough valid samples. Found {len(self.sample_indices)}, but need at least {offset + limit}")
        
        # Initialize lists for data
        set_y, set_vjp, set_eig = [], [], []
        
        indices = []
        
        # Load saturation states, VJPs, and eigenvectors for the specified subset
        for s_idx in self.sample_indices[offset:offset + limit]:
            # Determine the appropriate states file path based on index
            if s_idx <= 40:  # This logic should match your data organization
                states_path = os.path.join(self.states_dir, f'num_ev_8_stateonly/states_sample_{s_idx}.jld2')
            else:
                states_path = os.path.join(self.states_dir, f'num_ev_8/saturation_sample_{s_idx}.jld2')
            
            # Load states
            with h5py.File(states_path, 'r') as f1:
                if s_idx <= 40:
                    states_refs = f1['single_stored_object'][:]
                else:
                    states_refs = f1['sat_series'][:]
                
                states_tensors = []
                for ref in states_refs:
                    state_data = f1[ref][:]
                    state_tensor = torch.tensor(state_data, dtype=torch.float32)
                    states_tensors.append(state_tensor)
                
                set_y.append(torch.stack(states_tensors).reshape(self.nt, self.nx, self.ny))
            
            # Load VJPs and eigenvectors if required
            if self.vjp_dir and self.eigvec_dir:
                vjp_path = os.path.join(self.vjp_dir, f'FIM_vjp_sample_{s_idx}.jld2')
                eigvec_path = os.path.join(self.eigvec_dir, f'FIM_eigvec_sample_{s_idx}.jld2')
                
                with h5py.File(eigvec_path, 'r') as f2, h5py.File(vjp_path, 'r') as f3:
                    eigvec = f2['single_stored_object'][:]
                    vjp = f3['single_stored_object'][:]
                    
                    cur_vjp = torch.tensor(vjp, dtype=torch.float32).reshape(torch.tensor(vjp, dtype=torch.float32).shape[0], 8, self.nx, self.ny)[:, :self.num_vec]
                    cur_eig = torch.tensor(eigvec, dtype=torch.float32).reshape(torch.tensor(eigvec, dtype=torch.float32).shape[0], 8, self.nx, self.ny)[:, :self.num_vec]

                    set_vjp.append(cur_vjp[:self.nt]) 
                    set_eig.append(cur_eig[:self.nt])
                    
            indices.append(s_idx)
        
        # Process inputs
        x_raw = torch.tensor(self.set_x[offset:offset + limit], dtype=torch.float32)
        y_raw = torch.stack(set_y)
        
        # Reshape inputs
        x_raw = x_raw.unsqueeze(1)
        x_raw = x_raw.repeat(1, self.nt, 1, 1)
        
        # Process VJPs and eigenvectors if available
        vjp_data = None
        eigvec_data = None
        
        if set_vjp and set_eig:
            # Process VJPs
            vjp_data = torch.stack(set_vjp)
            vjp_data = vjp_data / torch.norm(vjp_data)
            
            # Process eigenvectors
            eigvec_data = torch.stack(set_eig)
            eigvec_data = eigvec_data / torch.norm(eigvec_data)

        # Add channel dimension to x and y
        x_raw = x_raw.unsqueeze(1)
        y_raw = y_raw.unsqueeze(1)
        
        return x_raw, y_raw, vjp_data, eigvec_data, indices
    
    def get_dataloader(self, offset: int, limit: int, shuffle: bool = True):
        """
        Get a DataLoader for a specific subset of the data.
        
        Args:
            offset: Starting index for the data subset
            limit: Number of samples to include
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the specified subset
        """
        # Load data subset
        x_raw, y_raw, vjp_data, eigvec_data, indices = self._load_data_subset(offset, limit)
        
        # Create dataset
        dataset = CustomDataset(x_raw, y_raw, vjp_data, eigvec_data, indices)
        
        # Create and return dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
