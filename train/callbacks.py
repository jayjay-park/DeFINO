import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
import matplotlib.colors as colors

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


class BaseVisualizationCallback(Callback):
    """Base class for visualization callbacks."""
    
    def __init__(self, output_dir=None, save_to_disk=False, log_to_neptune=False, num_plots=1):
        """
        Initialize visualization callback.
        
        Args:
            output_dir: Directory to save visualizations locally
            save_to_disk: Whether to save to disk
            log_to_neptune: Whether to log to Neptune
        """
        super().__init__()
        self.output_dir = output_dir
        self.save_to_disk = save_to_disk
        self.num_plots = num_plots
        if output_dir and not os.path.exists(output_dir) and save_to_disk:
            os.makedirs(output_dir, exist_ok=True)
        
        self.log_to_neptune = log_to_neptune and NEPTUNE_AVAILABLE
    
    def save_figure(self, fig, filename):
        """Save figure locally and/or to Neptune."""
        if self.save_to_disk and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
        return fig
    
    def log_figure(self, trainer, pl_module, fig, tag, step=None):
        """Log figure to Neptune if available."""
        if self.log_to_neptune and hasattr(trainer.logger, 'experiment'):
            if step is None:
                step = trainer.global_step
            
            # Log to Neptune
            trainer.logger.experiment[f"visualizations/{tag}"].append(fig)


class SaturationVisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing saturation predictions."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        self.plot_saturation_results(trainer, pl_module, trainer.train_dataloader, "train")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():
            self.plot_saturation_results(trainer, pl_module, trainer.val_dataloaders, "val")
    
    def plot_saturation_results(self, trainer, pl_module, dataloader, tag):
        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            # Get model predictions
            x = batch['x'].to(pl_module.device)
            y_true = batch['y']
            
            with torch.no_grad():
                y_pred = pl_module(x)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            channel_idx = 0
            
            # # Plot for each timestep
            # for timestep in range(y_true.shape[1]):
            #     fig = self.plot_results(
            #         y_true[batch_idx, channel_idx, timestep].squeeze().cpu().numpy(),
            #         y_pred[batch_idx, channel_idx, timestep].squeeze().detach().cpu().numpy(),
            #         f"Timestep {timestep+1}"
            #     )
            #     filename = f"saturation_timestep_{timestep+1}_epoch_{trainer.current_epoch}.png"
            #     self.save_figure(fig, filename)
            #     self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/saturation/timestep_{timestep+1}", trainer.current_epoch)
            #     plt.close(fig)

            # Plot all timesteps in a single figure
            fig = self.plot_multiple_timesteps(
                y_true[batch_idx, channel_idx].squeeze().cpu().numpy(),
                y_pred[batch_idx, channel_idx].squeeze().detach().cpu().numpy()
            )
            
            filename = f"saturation_{tag}_timesteps_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/saturation_all", trainer.current_epoch)
            plt.close(fig)
    
    def plot_results(self, true1, pred1, title_suffix=""):
        """
        Plot true saturation, predicted saturation, and error.
        
        Args:
            true1: True saturation field
            pred1: Predicted saturation field
            title_suffix: Suffix for plot title
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(19, 5))
        plt.rcParams.update({'font.size': 14})
        
        # True saturation
        plt.subplot(1, 3, 1)
        plt.imshow(true1.numpy(), cmap='jet', vmin=0.0, vmax=1.0)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"True Saturation {title_suffix}")
        
        # Predicted saturation
        plt.subplot(1, 3, 2)
        plt.imshow(pred1.numpy(), cmap='jet', vmin=0.0, vmax=1.0)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Predicted Saturation {title_suffix}")
        
        # Error
        plt.subplot(1, 3, 3)
        error = np.abs(true1.numpy() - pred1.numpy())
        vmax_error = error.max()
        plt.imshow(error, cmap='magma', vmin=0.0, vmax=vmax_error)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('Error')
        
        return fig
    
    def plot_multiple_timesteps(self, true, pred, cmap='jet'):
        """
        Plot multiple timesteps in a single figure.
        
        Args:
            true: True saturation fields (timesteps, height, width)
            pred: Predicted saturation fields (timesteps, height, width)
            cmap: Colormap to use
            
        Returns:
            matplotlib figure
        """
        num_timesteps = true.shape[0]
        fig, axes = plt.subplots(3, num_timesteps, figsize=(4*num_timesteps, 8))
        plt.rcParams.update({'font.size': 12})
        
        # Plot true saturation (top row)
        for t in range(num_timesteps):
            im = axes[0, t].imshow(true[t], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[0, t].set_title(f'True t={t+1}')
            if t == 0:
                axes[0, t].set_ylabel('True')
            fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
        
        # Plot predicted saturation (bottom row)
        for t in range(num_timesteps):
            im = axes[1, t].imshow(pred[t], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[1, t].set_title(f'Pred t={t+1}')
            if t == 0:
                axes[1, t].set_ylabel('Predicted')
            fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.04)
            
        # Plot error (third row)
        for t in range(num_timesteps):
            im = axes[2, t].imshow(np.abs(true[t] - pred[t]), cmap='magma', vmin=0.0, vmax=1.0)
            axes[2, t].set_title(f'Error t={t+1}')
            if t == 0:
                axes[2, t].set_ylabel('Error')
            fig.colorbar(im, ax=axes[2, t], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig


class JacobianVisualizationCallback(BaseVisualizationCallback):
    """Callback for visualizing vector-Jacobian products."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each training epoch."""
        self.plot_jacobian_results(trainer, pl_module, trainer.train_dataloader, "train")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Create visualization at the end of each validation epoch."""
        with torch.no_grad():   
            self.plot_jacobian_results(trainer, pl_module, trainer.val_dataloaders, "val")
    
    def plot_jacobian_results(self, trainer, pl_module, dataloader, tag):
        plot_counter = 0
        for batch in dataloader:
            if plot_counter >= self.num_plots:
                break
            plot_counter += 1

            # Only perform Jacobian visualization if model has VJP computation
            if not hasattr(pl_module, 'compute_vjps'):
                return
            
            x = batch['x'].to(pl_module.device)
            eigvec = batch['eigvec'].to(pl_module.device)
            true_vjp = batch['vjp'].to(pl_module.device)
            
            # Skip if no eigenvectors in batch
            if 'eigvec' not in batch or 'vjp' not in batch:
                trainer.logger.experiment[f"errors/{tag}"].append(f"No eigenvectors or VJPs in batch")
                return

            # Compute predicted VJPs
            pred_vjp = pl_module.compute_vjps(x, eigvec, clear_memory=True)
            
            # Create visualizations
            batch_idx = 0  # Visualize first sample in batch
            vec_idx = 0    # Visualize first vector
            
            # # Plot for each timestep
            # for timestep in range(eigvec.shape[1]):
            #     # Plot VJP results for current timestep
            #     fig = self.plot_vjp_results(
            #         eigvec[batch_idx, timestep, vec_idx].detach().cpu(),
            #         true_vjp[batch_idx, timestep, vec_idx].detach().cpu(),
            #         pred_vjp[batch_idx, timestep, vec_idx].detach().cpu(),
            #         f"Timestep {timestep+1}"
            #     )
            #     filename = f"vjp_timestep_{timestep+1}_epoch_{trainer.current_epoch}.png"
            #     self.save_figure(fig, filename)
            #     self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/jacobian/timestep_{timestep+1}", trainer.current_epoch)
            #     plt.close(fig)
                
            # Plot all timesteps in a single figure
            fig = self.plot_multiple_timesteps(
                eigvec[batch_idx, :, vec_idx].squeeze().cpu().numpy(),
                true_vjp[batch_idx, :, vec_idx].squeeze().cpu().numpy(),
                pred_vjp[batch_idx, :, vec_idx].squeeze().detach().cpu().numpy()
            )
            
            filename = f"jacobian_{tag}_timesteps_sample_{batch['idx'][batch_idx]}_epoch_{trainer.current_epoch}.png"
            self.save_figure(fig, filename)
            self.log_figure(trainer, pl_module, fig, f"{tag}/sample_{batch['idx'][batch_idx]}/jacobian_all", trainer.current_epoch)
            plt.close(fig)
    
    def plot_multiple_timesteps(self, eigvec, true_vjp, pred_vjp, cmap='seismic'):
        """
        Plot multiple timesteps in a single figure.
        
        Args:
            eigvec: Eigenvectors (timesteps, height, width)
            true_vjp: True vector-Jacobian products (timesteps, height, width)
            pred_vjp: Predicted vector-Jacobian products (timesteps, height, width)
            cmap: Colormap to use
            
        Returns:
            matplotlib figure
        """
        num_timesteps = eigvec.shape[0]
        fig, axes = plt.subplots(4, num_timesteps, figsize=(4*num_timesteps, 12))
        plt.rcParams.update({'font.size': 10})
        
        # Plot eigenvector
        for t in range(num_timesteps):
            eigvec_np = eigvec[t]
            eigvec_range = max(abs(eigvec_np.min()), abs(eigvec_np.max()))
            norm = SymLogNorm(linthresh=0.1 * eigvec_range, vmin=-eigvec_range, vmax=eigvec_range)
            im = axes[0, t].imshow(eigvec_np, cmap=cmap, norm=norm)
            axes[0, t].set_title(f'Eigenvector t={t+1}')
            if t == 0:
                axes[0, t].set_ylabel('Eigenvector')
            fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)
        
        # Plot true vJp
        for t in range(num_timesteps):
            true_vjp_np = true_vjp[t]
            true_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max()))
            norm_true = SymLogNorm(linthresh=0.1 * true_range, vmin=-true_range, vmax=true_range)
            im = axes[1, t].imshow(true_vjp_np, cmap=cmap, norm=norm_true)
            axes[1, t].set_title(f'True VJP t={t+1}')
            if t == 0:
                axes[1, t].set_ylabel('True VJP')
            fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.04)
            
        # Plot predicted vJp
        for t in range(num_timesteps):
            pred_vjp_np = pred_vjp[t]
            pred_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max())) # scale according to true vjp
            norm_pred = SymLogNorm(linthresh=0.1 * pred_range, vmin=-pred_range, vmax=pred_range)
            im = axes[2, t].imshow(pred_vjp_np, cmap=cmap, norm=norm_pred)
            axes[2, t].set_title(f'Predicted VJP t={t+1}')
            if t == 0:
                axes[2, t].set_ylabel('Predicted VJP')
            fig.colorbar(im, ax=axes[2, t], fraction=0.046, pad=0.04)
            
        # Plot error
        for t in range(num_timesteps):
            error_np = np.abs(true_vjp[t] - pred_vjp[t])
            error_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max())) # scale according to true vjp
            norm_error = SymLogNorm(linthresh=0.1 * error_range, vmin=-error_range, vmax=error_range)
            im = axes[3, t].imshow(error_np, cmap='magma', norm=norm_error)
            axes[3, t].set_title(f'Error t={t+1}')
            if t == 0:
                axes[3, t].set_ylabel('Error')
            fig.colorbar(im, ax=axes[3, t], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def plot_vjp_results(self, vector, true_vjp, pred_vjp, title_suffix=""):
        """
        Plot vector, true VJP, and predicted VJP.
        
        Args:
            vector: Input vector
            true_vjp: True vector-Jacobian product
            pred_vjp: Predicted vector-Jacobian product
            title_suffix: Suffix for plot title
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(15, 5))
        plt.rcParams.update({'font.size': 14})
        
        # Input vector
        plt.subplot(1, 3, 1)
        vector_np = vector.numpy()
        vector_range = max(abs(vector_np.min()), abs(vector_np.max()))
        norm = SymLogNorm(linthresh=0.1 * vector_range, vmin=-vector_range, vmax=vector_range)
        plt.imshow(vector_np, cmap='seismic', norm=norm)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Vector {title_suffix}")
        
        # True VJP
        plt.subplot(1, 3, 2)
        true_vjp_np = true_vjp.numpy()
        true_range = max(abs(true_vjp_np.min()), abs(true_vjp_np.max()))
        norm_true = SymLogNorm(linthresh=0.1 * true_range, vmin=-true_range, vmax=true_range)
        plt.imshow(true_vjp_np, cmap='seismic', norm=norm_true)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"True VJP {title_suffix}")
        
        # Predicted VJP
        plt.subplot(1, 3, 3)
        pred_vjp_np = pred_vjp.numpy()
        pred_range = max(abs(pred_vjp_np.min()), abs(pred_vjp_np.max()))
        norm_pred = SymLogNorm(linthresh=0.1 * pred_range, vmin=-pred_range, vmax=pred_range)
        plt.imshow(pred_vjp_np, cmap='seismic', norm=norm_pred)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f"Predicted VJP {title_suffix}")
        
        return fig
