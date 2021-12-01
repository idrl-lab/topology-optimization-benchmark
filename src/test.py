"""
Runs a model on a single node across multiple gpus.
"""
import os
from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl

from src.LayoutDeepRegression import Model


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        precision=16 if hparams.use_16bit else 32,
        # limit_test_batches=0.05
    )

    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    model_path = list(Path(model_path).glob("*.ckpt"))[0]
    test_model = model.load_from_checkpoint(checkpoint_path=model_path, hparams=hparams)

    # ------------------------
    # 3 START PREDICTING
    # ------------------------
    print(hparams)
    print()

    trainer.test(model=test_model)


if __name__ == "__main__":

    # ------------------------
    # TESTING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    config_path = Path(__file__).absolute().parent / "config/config.yml"
    parser = configargparse.ArgParser(default_config_files=[str(config_path)], description="Hyper-parameters.")
    parser.add_argument("--config", is_config_file=True, default=False, help="config file path")

    # args
    parser.add_argument("--save_check_num", default=0, type=int, help="checkpoint for test")
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--max_iters", default=40000, type=int)
    parser.add_argument("--resume_from_checkpoint", type=str, help="resume from checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--gpus", type=int, default=0, help="how many gpus")
    parser.add_argument("--use_16bit", type=bool, default=False, help="use 16bit precision")
    parser.add_argument("--val_check_interval", type=float, default=1,
                        help="how often within one training epoch to check the validation set")
    parser.add_argument("--profiler", action="store_true", help="use profiler")
    parser.add_argument("--test_args", action="store_true", help="print args")

    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # test args in cli
    if hparams.test_args:
        print(hparams)
    else:
        main(hparams)
