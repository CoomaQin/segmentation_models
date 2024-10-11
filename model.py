import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch


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
            obj_pred_mask, obj_mask, mode="multiclass", num_classes=self.number_of_classes
        )
        atp, afp, afn, atn = smp.metrics.get_stats(
            area_pred_mask, area_mask, mode="multiclass", num_classes=self.number_of_classes
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