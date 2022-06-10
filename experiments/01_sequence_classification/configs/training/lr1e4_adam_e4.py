import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


def configure_optimizers(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return optimizer


def optimizer_step(
    epoch=None,
    batch_idx=None,
    optimizer=None,
    optimizer_idx=None,
    optimizer_closure=None,
    on_tpu=None,
    using_native_amp=None,
    using_lbfgs=None
):
    optimizer.step(closure=optimizer_closure)
    optimizer.zero_grad()



config = dict(

    # Batching args:
    train_batch_size=32,
    validation_batch_size=64,
    padding_strategy="max_length",
    num_workers=8,

    # Optimization args:
    configure_optimizers=configure_optimizers,
    optimizer_step=optimizer_step,
    enable_checkpointing=True,
    min_epochs=1,
    max_epochs=1,
    accelerator="cpu",
    devices=1,

    # Logging args:
    val_check_interval=200,
    log_every_n_steps=50
)