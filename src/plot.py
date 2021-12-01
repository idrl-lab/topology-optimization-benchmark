"""
Runs a model on a single node across multiple gpus.
"""
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse

from src.LayoutDeepRegression import Model


def main(hparams):
    model = Model(hparams).cuda()

    print(hparams)
    print()

    # Model loading
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    print(ckpt)

    model = model.load_from_checkpoint(str(ckpt))

    model.eval()
    model.cuda()
    mae_test = []

    # Testing Set
    root = hparams.data_root
    test_list = hparams.test_list
    file_path = os.path.join(root, test_list)
    root_dir = os.path.join(root, 'test', 'test')

    with open(file_path, 'r') as fp:
        for line in fp.readlines():
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_true, layout = data["layout_input"], data["xData_output"]

            # Plot Layout and Real Temperature Field
            fig = plt.figure(figsize=(10.5, 3))

            grid_x = np.linspace(0, 0.1, num=40)
            grid_y = np.linspace(0, 0.1, num=40)
            X, Y = np.meshgrid(grid_x, grid_y)

            plt.subplot(131)
            plt.title('Heat Source Layout')
            im = plt.pcolormesh(X, Y, layout)
            plt.colorbar(im)
            fig.tight_layout(w_pad=3.0)

            layout = torch.Tensor(layout / 1000.0).unsqueeze(0).unsqueeze(0).cuda()
            print(layout.size())
            heat = torch.Tensor((u_true - 298) / 50.0).unsqueeze(0).unsqueeze(0).cuda()
            with torch.no_grad():
                heat_pre = model(layout)
                mae = F.l1_loss(heat, heat_pre) * 50
                print('MAE:', mae)
            mae_test.append(mae.item())
            heat_pre = heat_pre.squeeze(0).squeeze(0).cpu().numpy() * 50.0 + 298
            hmax = max(np.max(heat_pre), np.max(u_true))
            hmin = min(np.min(heat_pre), np.min(u_true))

            plt.subplot(132)
            plt.title('Real Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, u_true, vmin=hmin, vmax=hmax)
                plt.axis('equal')
            else:
                im = plt.pcolormesh(X, Y, u_true, vmin=hmin, vmax=hmax)
            plt.colorbar(im)

            plt.subplot(133)
            plt.title('Predicted Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, heat_pre, vmin=hmin, vmax=hmax)
                plt.axis('equal')
            else:
                im = plt.pcolormesh(X, Y, heat_pre, vmin=hmin, vmax=hmax)
            plt.colorbar(im)

            save_name = os.path.join('outputs/predict_plot', os.path.splitext(os.path.basename(path))[0]+'.png')
            fig.savefig(save_name, dpi=300)
            plt.close()

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt('outputs/mae_test.csv', mae_test, fmt='%f', delimiter=',')


if __name__ == "__main__":

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    # default configuration file
    config_path = Path(__file__).absolute().parent / "config/config.yml"
    parser = configargparse.ArgParser(default_config_files=[str(config_path)], description="Hyper-parameters.")

    # configuration file
    parser.add_argument("--config", is_config_file=True, default=False, help="config file path")

    # mode
    parser.add_argument("-m", "--mode", type=str, default="train", help="model: train or test or plot")

    # args for training
    parser.add_argument("--gpus", type=int, default=0, help="how many gpus")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--lr", default="0.01", type=float)
    parser.add_argument("--resume_from_checkpoint", type=str, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=2, type=int, help="num_workers in DataLoader")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--use_16bit", type=bool, default=False, help="use 16bit precision")
    parser.add_argument("--profiler", action="store_true", help="use profiler")

    # args for validation
    parser.add_argument("--val_check_interval", type=float, default=1,
                        help="how often within one training epoch to check the validation set")

    # args for testing
    parser.add_argument("--test_check_num", default='0', type=str, help="checkpoint for test")
    parser.add_argument("--test_args", action="store_true", help="print args")

    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # test args in cli
    if hparams.test_args:
        print(hparams)
    else:
        main(hparams)
