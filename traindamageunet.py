import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import cv2

from glob import glob

import pytorch_lightning as pl
from pytorch_toolbelt.losses.lovasz import _lovasz_softmax
from pytorch_toolbelt.losses.focal import FocalLoss
from pytorch_toolbelt.losses.dice import DiceLoss

from albumentations import (
    PadIfNeeded,
    ISONoise,
    RandomFog,
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
        RandomFog(fog_coef_lower=0.025, fog_coef_upper=0.1, p=0.5),
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
        self.post = [fn.replace("pre_disaster", "post_disaster") for fn in self.pre]
        self.prey = [
            fn.replace("_disaster", "_mask_disaster").replace("images", "targets")
            for fn in self.pre
        ]
        self.posty = [
            fn.replace("_disaster", "_mask_disaster").replace("images", "targets")
            for fn in self.post
        ]

    def __len__(self):
        return len(self.pre)

    def __getitem__(self, idx):
        [pre, post, prey, posty] = [
            cv2.imread(f[idx]) for f in [self.pre, self.post, self.prey, self.posty]
        ]
        if self.aug:
            preyposty = np.concatenate([prey, posty], axis=2)
            augmented = transformaug(
                image=pre,
                postimage=post,
                mask=preyposty,
                additional={"postimage": "image"},
            )
            pre, post, prey, posty = (
                augmented["image"],
                augmented["postimage"],
                augmented["mask"][:, :, 0:3],
                augmented["mask"][:, :, 3:6],
            )
        loc = prey[:, :, 0].astype(np.int64) // 63
        loc = torch.from_numpy(loc)
        locedge = prey[:, :, 1].astype(np.int64) // 255
        locedge = torch.from_numpy(locedge)
        dmg = posty[:, :, 0].astype(np.int64) // 63
        dmg = torch.from_numpy(dmg)

        pre, post = transformimg(pre), transformimg(post)
        return pre, post, locedge, dmg


from effdamageunet import DamageUNet


class XViewSystem(pl.LightningModule):
    def __init__(self, hparams=None):
        super(XViewSystem, self).__init__()
        self.xviewmodel = DamageUNet()
        self.focal = FocalLoss(alpha=0.5, gamma=2, ignore_index=0)
        self.dice = DiceLoss("multiclass", classes=[1, 2, 3, 4])
        self.crossentropy = nn.CrossEntropyLoss(
            weight=torch.Tensor([0.0, 0.1, 1.0, 1.0, 1.0]), ignore_index=0
        )

    def forward(self, x):
        return self.xviewmodel(x)

        # learning rate warm-up

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        # warm up lr
        if self.trainer.global_step < 100:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 100.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.01
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):

        pre, post, loc, dmg = batch
        locy, dmgy = self.forward([pre, post])
        dmgce = self.crossentropy(dmgy, dmg)
        dmgdice = self.dice(dmgy, dmg)
        dmgfocal = self.focal(dmgy, dmg)

        loss = dmgce + dmgdice + dmgfocal

        tensorboardlog = {
            "loss": loss,
            "dmgce": dmgce,
            "dmgdice": dmgdice,
            "dmgfocal": dmgfocal,
        }
        return {"loss": loss, "log": tensorboardlog}

    def validation_step(self, batch, batch_idx):

        pre, post, loc, dmg = batch
        locy, dmgy = self.forward([pre, post])
        dmgce = self.crossentropy(dmgy, dmg)
        dmgdice = self.dice(dmgy, dmg)
        dmgfocal = self.focal(dmgy, dmg)

        dmgloss = dmgce + dmgdice + dmgfocal

        val_loss = dmgloss

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
            XViewDataset(), shuffle=True, drop_last=True, num_workers=16, batch_size=32
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            XViewDataset(600), shuffle=True, drop_last=True, num_workers=8, batch_size=8
        )

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            XViewDataset(64), shuffle=True, drop_last=True, num_workers=8, batch_size=8
        )


if __name__ == "__main__":
    model = XViewSystem(None)
    trainer = pl.Trainer(
        default_save_path="workspace/damage",
        check_val_every_n_epoch=5,
        min_nb_epochs=210,
        max_nb_epochs=210,
        gpus=1,
        use_amp=True,
    )
    trainer.fit(model)
