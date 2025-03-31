import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from model import GCSModel
from dataset import GCSDataLoader
from callbacks import SaturationVisualizationCallback, JacobianVisualizationCallback
from utils import (
    load_config, save_config, setup_logging, setup_neptune_logging,
    create_directories, log_config_to_file, set_seed
)

try:
    from pytorch_lightning.loggers import NeptuneLogger
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False


def main():
    """Main function to run the training process."""
    parser = argparse.ArgumentParser(description='GCS FNO Training with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/mse.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment directories
    directories = create_directories(config)
    
    # Set up logging
    log_file, logger = setup_logging(
        log_dir=directories['log_dir'],
        experiment_name=directories['experiment']
    )
    
    # Log configuration
    log_config_to_file(config, logger)
    
    # Save configuration to experiment directory
    save_config(config, os.path.join(directories['output_dir'], 'config.yaml'))
    
    # Set random seed for reproducibility
    set_seed(config.experiment.seed)
    
    # Set up Neptune logging if enabled
    neptune_run = setup_neptune_logging(config)
    
    # Create train loader
    data_loader = GCSDataLoader(
        k_file=config.data.k_file,
        states_dir=config.data.states_dir,
        vjp_dir=config.data.vjp_dir,
        eigvec_dir=config.data.eigvec_dir,
        batch_size=config.training.batch_size,
        nt=config.data.nt,
        nx=config.data.nx,
        ny=config.data.ny,
        num_vec=config.data.num_vec,
        num_workers=config.training.num_workers
    )
    
    train_loader = data_loader.get_dataloader(offset=0, limit=config.data.num_train, shuffle=False) # TODO: figure out how to set to true & have same plotting in callback
    test_loader = data_loader.get_dataloader(offset=config.data.num_train, limit=config.data.num_test, shuffle=False)

    # Create model
    model = GCSModel(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        decoder_layer_size=config.model.decoder_layer_size,
        num_fno_layers=config.model.num_fno_layers,
        num_fno_modes=config.model.num_fno_modes,
        padding=config.model.padding,
        dimension=config.model.dimension,
        latent_channels=config.model.latent_channels,
        loss_type=config.training.loss_type,
        reg_param=config.training.reg_param,
        scale_factor=config.training.scale_factor,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Set up loggers
    loggers = []
    
    # # TensorBoard logger
    # tb_logger = TensorBoardLogger(
    #     save_dir=directories['log_dir'],
    #     name="tensorboard",
    #     version=directories['experiment']
    # )
    # loggers.append(tb_logger)
    
    # Neptune logger
    if NEPTUNE_AVAILABLE and config.neptune.enabled:
        neptune_logger = NeptuneLogger(
            run=neptune_run,
            log_model_checkpoints=config.training.log_checkpoint
        )
        loggers.append(neptune_logger)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=directories['checkpoint_dir'],
        filename=f"GCS_{{epoch:03d}}_{{val_rel_l2_loss:.4f}}",
        monitor="val_rel_l2_loss",
        save_top_k=config.training.save_top_k,
        mode="min",
        save_last=True,
        every_n_epochs=config.training.checkpoint_interval
    )
    if config.training.enable_checkpointing:
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_rel_l2_loss",
        patience=20,
        min_delta=1e-4,
        mode="min"
    )
    callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # Visualization callbacks
    saturation_viz_callback = SaturationVisualizationCallback(
        output_dir=directories['plot_sat_dir'],
        log_to_neptune=config.visualization.log_to_neptune,
        save_to_disk=config.visualization.save_to_disk,
        num_plots=config.visualization.num_plots
    )
    callbacks.append(saturation_viz_callback)
    
    jacobian_viz_callback = JacobianVisualizationCallback(
        output_dir=directories['plot_jac_dir'],
        log_to_neptune=config.visualization.log_to_neptune,
        save_to_disk=config.visualization.save_to_disk,
        num_plots=config.visualization.num_plots
    )
    callbacks.append(jacobian_viz_callback)

    # Create trainer and set checkpointing to false
    trainer = pl.Trainer(
        max_epochs=config.training.num_epoch,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        accumulate_grad_batches=config.training.accum_steps,
        deterministic=True,
        enable_checkpointing=config.training.enable_checkpointing,
    )
    
    # Start training
    logger.info(f"Starting training with {config.training.loss_type} loss")
    trainer.fit(model, train_loader, test_loader)
    
    # Save best model path
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")
    
    # Close Neptune run if it was initialized
    if neptune_run:
        neptune_run.stop()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()
