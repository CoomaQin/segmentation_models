import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import cv2
import pytorch_lightning as pl
import torch
import albumentations as A
from model import CamVidModel


class Dataset(BaseDataset):
    AREA_CLASSES = [
        "unlabelled",
        "restricted",
        "ballast",
        "track"
    ]

    OBJ_CLASSES = [
        "unlabelled",
        "person"
    ]

    def __init__(self, data_root, annotation_path, height=640, width=640, classes=None, augmentation=None):
        self.images_fps = []
        self.masks_fps = []
        # read lines from a file
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            image_name, mask_name = line.strip().split()
            self.images_fps.append(os.path.join(data_root, image_name))
            self.masks_fps.append(os.path.join(data_root, mask_name))
        self.height, self.width = height, width

        # Always map background ('unlabelled') to 0
        self.background_class = self.OBJ_CLASSES.index("unlabelled")

        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.OBJ_CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.OBJ_CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}
        self.class_map.update(
            {
                v: i
                for i, v in enumerate(self.class_values)
            }
        )

        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_LANCZOS4)

        # area
        mask = cv2.imread(self.masks_fps[i])
        area_mask = mask[:, :, 0]
        obj_mask = mask[:, :, 0]
        obj_mask = cv2.resize(obj_mask, (self.height, self.width), interpolation=cv2.INTER_NEAREST)
        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(obj_mask)
        # Remap the mask according to the dynamically created class map
        for class_value, new_value in self.class_map.items():
            mask_remap[obj_mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.images_fps)


# training set images augmentation
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=640, min_width=640, always_apply=True),
        A.RandomCrop(height=640, width=640, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    DATA_DIR = "data/merge"
    train_dataset = Dataset(
        DATA_DIR,
        "data/merge/dummy.txt",
        augmentation=get_training_augmentation(),
    )

    valid_dataset = Dataset(
         DATA_DIR,
        "data/merge/test.txt",
        augmentation=get_validation_augmentation(),
    )
    # Change to > 0 if not on Windows machine
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=3)

    # Some training hyperparameters
    EPOCHS = 5
    T_MAX = EPOCHS * len(train_loader)
    # Always include the background as a class
    OUT_CLASSES = len(train_dataset.OBJ_CLASSES)

    model = CamVidModel("DeepLabV3", "resnext50_32x4d", in_channels=3, out_classes=OUT_CLASSES)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, devices=2, accelerator="gpu")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)