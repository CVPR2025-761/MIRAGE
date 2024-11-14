import pytorch_lightning as pl
from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lars import LARS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PretrainModel(pl.LightningModule):
    def __init__(self, text_encoder, image_encoder, train_dataset, validation_dataset=None, test_dataset=None, embed_dim=768, num_workers=16,  gpus=[0], exclude_bn_bias = False, max_epochs=20, warmup_epochs=2, batch_size=16, optim='adam', scheduler='linear_warmup_cosine_annealing', learning_rate=1e-4, learning_rate_start=1e-6, learning_rate_end=1e-5, weight_decay=1e-6, frozen_text_encoder=False, save_each_epoch=False):
        super().__init__()

        # define encoders & modules
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vision_width =  image_encoder.get_width()
        self.text_width = text_encoder.get_width()
        self.embed_dim = embed_dim
       
        # Define hyper-params for optimization
        self.exclude_bn_bias = exclude_bn_bias
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.weight_decay= weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # define dataset & loader
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        if self.train_dataset is not None:
            self.train_iters_per_epoch = len(self.train_dataset)  // (len(gpus) * batch_size)
            print(self.train_iters_per_epoch)

        self.gpus = gpus
        # tuning and params
        if frozen_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        #self.inititalize_parameters()
        self.save_each_epoch = save_each_epoch

    def on_train_epoch_end(self):
        if self.save_each_epoch:
            torch.save(self.state_dict(), f"epoch_{self.current_epoch}.pth")
        return super().on_train_epoch_end()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) :
        if self.validation_dataset is not None:
            return DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
    def text_dataloader(self) :
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_text(self, text):
        return self.text_encoder(text)
    
    def configure_optimizers(self):
        max_epochs = self.max_epochs
        warmup_epochs = self.warmup_epochs
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
    
        warmup_steps = self.train_iters_per_epoch * warmup_epochs 
        self.warmup_steps = warmup_steps
        total_steps = self.train_iters_per_epoch * max_epochs
        if self.scheduler == 'cosine_warmup_linear_annealing':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'linear_warmup_cosine_annealing':
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=warmup_steps,
                    max_epochs=total_steps,
                    warmup_start_lr=self.learning_rate_start, eta_min=self.learning_rate_end),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'cosine_decay':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps),
                "interval": "step",
                "frequency": 1,
            }
        return [optimizer], [scheduler]

    def shared_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, 'training')
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=True)
        return loss_dict['training_loss']
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, 'validation')
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['validation_loss']
    
    def test_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, 'test')
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['test_loss']