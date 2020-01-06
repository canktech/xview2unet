import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import cv2

from glob import glob

import pytorch_lightning as pl
from pytorch_toolbelt.losses.focal import FocalLoss
from pytorch_toolbelt.losses.dice import DiceLoss

from albumentations import (
    PadIfNeeded,
    ISONoise,
    RandomFog,  # coeff < 0.2 may work perhaps 0.1
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
)

transformimg = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406][::-1], std=[0.225, 0.224, 0.225][::-1]
        ),
    ]
)

transformaug = Compose(
    [
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ISONoise(p=0.5),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.5),
        RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.5),
    ]
)


class XViewDataset(Dataset):
    def __init__(
        self, size=None, aug=True, pattern="data/train/images1024/*pre_disaster*.png"
    ):
        self.name = "train"
        self.aug = aug
        self.pre = glob(pattern)

        if size:
            self.pre = self.pre[:size]
        self.prey = [
            fn.replace("_disaster", "_mask_disaster").replace("images", "targets")
            for fn in self.pre
        ]

    def __len__(self):
        return len(self.pre)

    def __getitem__(self, idx):
        [pre, prey] = [cv2.imread(f[idx]) for f in [self.pre, self.prey]]
        if self.aug:
            augmented = transformaug(image=pre, mask=prey)
            pre, prey = augmented["image"], augmented["mask"]
        loc = prey[:, :, 0].astype(np.int64) // 255
        loc = torch.from_numpy(loc)
        locedge = prey[:, :, 1].astype(np.int64) // 255
        locedge = torch.from_numpy(locedge)
        pre = transformimg(pre)
        return pre, loc, locedge


from efflocunet import LocUNet


class XViewLocSystem(pl.LightningModule):
    def __init__(self, hparams=None):
        super(XViewLocSystem, self).__init__()
        self.xviewmodel = LocUNet()
        self.focal = FocalLoss(alpha=0.5, gamma=2)
        self.dice = DiceLoss("multiclass")
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.xviewmodel(x)

        # learning rate warm-up

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        # warm up lr
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.01
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):

        pre, loc, locedge = batch
        locy = self.forward(pre)
        locce = self.cross_entropy(locy[:, 0:2, :, :], loc)
        locdice = self.dice(locy[:, 0:2, :, :], loc)
        locfocal = self.focal(locy[:, 0:2, :, :], loc)
        locedgece = self.cross_entropy(locy[:, 2:4, :, :], locedge)
        locedgedice = self.dice(locy[:, 2:4, :, :], locedge)
        locedgefocal = self.focal(locy[:, 2:4, :, :], locedge)

        locloss = locce + locdice + 5 * locfocal
        locedgeloss = locedgece + locedgedice + locedgefocal
        loss = locloss + 0.5 * locedgeloss

        tensorboardlog = {
            "loss": loss,
            "locce": locce,
            "locdice": locdice,
            "locfocal": locfocal,
            "locedgece": locedgece,
            "locedgedice": locedgedice,
            "locedgefocal": locedgefocal,
        }
        return {"loss": loss, "log": tensorboardlog}

    def validation_step(self, batch, batch_idx):

        pre, loc, locedge = batch
        locy = self.forward(pre)
        locce = self.cross_entropy(locy[:, 0:2, :, :], loc)
        locdice = self.dice(locy[:, 0:2, :, :], loc)
        locfocal = self.focal(locy[:, 0:2, :, :], loc)
        locedgece = self.cross_entropy(locy[:, 2:4, :, :], locedge)
        locedgedice = self.dice(locy[:, 2:4, :, :], locedge)
        locedgefocal = self.focal(locy[:, 2:4, :, :], locedge)

        locloss = locce + locdice + locfocal
        locedgeloss = locedgece + locedgedice + locedgefocal
        val_loss = locloss + 0.1 * locedgeloss

        return {
            "val_loss": val_loss,
        }

    def validation_end(self, outputs):
        avgloss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avgloss, "log": {"val_loss": avgloss}}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            XViewDataset(), shuffle=True, drop_last=True, num_workers=16, batch_size=64
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            XViewDataset(1200),
            shuffle=True,
            drop_last=True,
            num_workers=16,
            batch_size=16,
        )

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            XViewDataset(64), shuffle=True, drop_last=True, num_workers=8, batch_size=8
        )


if __name__ == "__main__":

    """
    model = XViewLocSystem.load_from_metrics(
            weights_path='lightning_logs/version_23/checkpoints/_ckpt_epoch_10.ckpt',
            tags_csv='lightning_logs/version_23/meta_tags.csv',
            )
    """

    model = XViewLocSystem()
    trainer = pl.Trainer(
        default_save_path="workspace/loc",
        check_val_every_n_epoch=5,
        min_nb_epochs=90,
        max_nb_epochs=90,
        gpus=1,
        use_amp=True,
    )
    trainer.fit(model)
