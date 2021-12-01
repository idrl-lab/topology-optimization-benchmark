# encoding: utf-8
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule

from src.data.layout import LayoutDataset
import src.utils.np_transforms as transforms
import src.models as models
from src.metric.metrics import Metric


class Model(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()
        self.criterion = nn.L1Loss()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_model(self):
        model_list = ["SegNet_AlexNet", "SegNet_VGG", "SegNet_ResNet18", "SegNet_ResNet50",
                      "SegNet_ResNet101", "SegNet_ResNet34", "SegNet_ResNet152",
                      "FPN_ResNet18", "FPN_ResNet50", "FPN_ResNet101", "FPN_ResNet34", "FPN_ResNet152",
                      "FCN_AlexNet", "FCN_VGG", "FCN_ResNet18", "FCN_ResNet50", "FCN_ResNet101",
                      "FCN_ResNet34", "FCN_ResNet152",
                      "UNet_VGG",
                      "VNet_user"]
        layout_model = self.hparams.model_name + '_' + self.hparams.backbone
        assert layout_model in model_list
        self.model = getattr(models, layout_model)()

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

    def __dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_layout]),
                    torch.tensor([self.hparams.std_layout]),
                ),
            ]
        )
        transform_heat = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_heat]),
                    torch.tensor([self.hparams.std_heat]),
                ),
            ]
        )

        # here only support format "mat"
        assert self.hparams.data_format == "mat"
        trainval_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.train_list,
            train=True,
            transform=transform_layout,
            target_transform=transform_heat,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.test_list,
            train=False,
            transform=transform_layout,
            target_transform=transform_heat,
        )

        # split train/val set
        train_length, val_length = int(len(trainval_dataset) * 0.8), int(len(trainval_dataset) * 0.2)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,
                                                                   [train_length, val_length])

        print(
            f"Prepared dataset, train:{int(len(train_dataset))},\
                val:{int(len(val_dataset))}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = self.__dataloader(train_dataset, shuffle=True)
        self.val_dataset = self.__dataloader(val_dataset, shuffle=False)
        self.test_dataset = self.__dataloader(test_dataset, shuffle=False)

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)
        self.log("train/training_mae", loss * self.hparams.std_heat)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(
                heat_pred[:4, ...], normalize=True
            )
            self.logger.experiment.add_image(
                "train_pred_heat_field", grid, self.global_step
            )
            if self.global_step == 0:
                grid = torchvision.utils.make_grid(
                    heat[:4, ...], normalize=True
                )
                self.logger.experiment.add_image(
                    "train_heat_field", grid, self.global_step
                )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val/val_mae", val_loss_mean.item() * self.hparams.std_heat)

    def test_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)

        data_config = Path(__file__).absolute().parent.parent / "config/data.yml"
        layout_metric = Metric(heat_pred, heat, boundary=self.hparams.boundary,
                             layout=layout, data_config=data_config, hparams=self.hparams)
        assert self.hparams.metric in layout_metric.metrics
        loss = getattr(layout_metric, self.hparams.metric)()
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss (" + self.hparams.metric +")", test_loss_mean.item())

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """Parameters you define here will be available to your model through `self.hparams`.
        """
        # dataset args
        parser.add_argument("--data_root", type=str, required=True, help="path of dataset")
        parser.add_argument("--train_list", type=str, required=True, help="path of train dataset list")
        parser.add_argument("--train_size", default=0.8, type=float, help="train_size in train_test_split")
        parser.add_argument("--test_list", type=str, required=True, help="path of test dataset list")
        parser.add_argument("--boundary", type=str, default="rm_wall", help="boundary condition")
        parser.add_argument("--data_format", type=str, default="mat", choices=["mat", "h5"], help="dataset format")

        # Normalization params
        parser.add_argument("--mean_layout", default=0, type=float)
        parser.add_argument("--std_layout", default=1, type=float)
        parser.add_argument("--mean_heat", default=0, type=float)
        parser.add_argument("--std_heat", default=1, type=float)

        # Model params (opt)
        parser.add_argument("--input_size", default=40, type=int)
        parser.add_argument("--model_name", type=str, default='SegNet', help="the name of chosen model")
        parser.add_argument("--backbone", type=str, default='ResNet18', help="the used backbone in the regression model")
        parser.add_argument("--metric", type=str, default='mae_global',
                            help="the used metric for evaluation of testing")
        return parser
