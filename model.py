import os
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


class BASModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, obj_classes, area_classes, **kwargs):
        super().__init__()
        self.obj_classes = obj_classes
        self.area_classes = area_classes
        out_classes = obj_classes + area_classes
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()
        area_mask, obj_mask = mask[:, :, :, 0], mask[:, :, :, 1]
        # Mask shape
        assert area_mask.ndim == 3 and obj_mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()
        obj_logits_mask = logits_mask[:, :self.obj_classes, :, :]
        area_logits_mask = logits_mask[:, self.obj_classes:, :, :]
        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        obj_loss = self.loss_fn(obj_logits_mask, obj_mask)
        area_loss = self.loss_fn(area_logits_mask, area_mask)

        # Apply softmax to get probabilities for multi-class segmentation
        obj_prob_mask = obj_logits_mask.softmax(dim=1)
        area_prob_mask = area_logits_mask.softmax(dim=1)
        # Convert probabilities to predicted class labels
        obj_pred_mask = obj_prob_mask.argmax(dim=1)
        area_pred_mask = area_prob_mask.argmax(dim=1)
        # Compute true positives, false positives, false negatives, and true negatives
        otp, ofp, ofn, otn = smp.metrics.get_stats(
            obj_pred_mask, obj_mask, mode="multiclass", num_classes=self.obj_classes
        )
        atp, afp, afn, atn = smp.metrics.get_stats(
            area_pred_mask, area_mask, mode="multiclass", num_classes=self.area_classes
        )

        return {
            "loss": obj_loss + area_loss,
            "otp": otp,
            "ofp": ofp,
            "ofn": ofn,
            "otn": otn,
            "atp": atp,
            "afp": afp,
            "afn": afn,
            "atn": atn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        otp = torch.cat([x["otp"] for x in outputs])
        ofp = torch.cat([x["ofp"] for x in outputs])
        ofn = torch.cat([x["ofn"] for x in outputs])
        otn = torch.cat([x["otn"] for x in outputs])
        atp = torch.cat([x["atp"] for x in outputs])
        afp = torch.cat([x["afp"] for x in outputs])
        afn = torch.cat([x["afn"] for x in outputs])
        atn = torch.cat([x["atn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_obj_iou = smp.metrics.iou_score(
            otp, ofp, ofn, otn, reduction="micro-imagewise"
        )
        dataset_obj_iou = smp.metrics.iou_score(otp, ofp, ofn, otn, reduction="micro")
        dataset_obj_f1 = smp.metrics.f1_score(otp, ofp, ofn, otn, reduction="micro")
        dataset_obj_miou = smp.metrics.iou_score(otp, ofp, ofn, otn, reduction="macro")
        dataset_obj_mf1 = smp.metrics.f1_score(otp, ofp, ofn, otn, reduction="macro")
        per_image_area_iou = smp.metrics.iou_score(
            atp, afp, afn, atn, reduction="micro-imagewise"
        )
        dataset_area_iou = smp.metrics.iou_score(atp, afp, afn, atn, reduction="micro")
        dataset_area_f1 = smp.metrics.f1_score(atp, afp, afn, atn, reduction="micro")
        dataset_area_miou = smp.metrics.iou_score(atp, afp, afn, atn, reduction="macro")
        dataset_area_mf1 = smp.metrics.f1_score(atp, afp, afn, atn, reduction="macro")
        metrics = {
            f"{stage}_per_image_obj_iou": per_image_obj_iou,
            f"{stage}_dataset_obj_iou": dataset_obj_iou,
            f"{stage}_dataset_obj_f1": dataset_obj_f1,
            f"{stage}_dataset_obj_miou": dataset_obj_miou,
            f"{stage}_dataset_obj_mf1": dataset_obj_mf1,
            f"{stage}_per_image_area_iou": per_image_area_iou,
            f"{stage}_dataset_area_iou": dataset_area_iou,
            f"{stage}_dataset_area_f1": dataset_area_f1,
            f"{stage}_dataset_area_miou": dataset_area_miou,
            f"{stage}_dataset_area_mf1": dataset_area_mf1
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes,  **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_f1": dataset_f1
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }