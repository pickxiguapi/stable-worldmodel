"""Train hierarchical GCHIQL with a chunk-conditioned low-level critic."""

import hydra
import lightning as pl
import stable_pretraining as spt
from gchiql import (
    SaveCkptCallback,
    get_data,
    get_gchiql_chunk_model,
    setup_pl_logger,
)

import stable_worldmodel as swm


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='gchiql_chunk',
)
def run(cfg):
    """Train GCHIQL-Chunk jointly from offline trajectories."""
    data = get_data(cfg)
    model = get_gchiql_chunk_model(cfg)
    callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=cfg.get('checkpoint_interval', 3),
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[callback],
        num_sanity_val_steps=1,
        logger=setup_pl_logger(cfg),
        enable_checkpointing=True,
    )
    cache_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    checkpoint_path = cache_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        ckpt_path=checkpoint_path if checkpoint_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
