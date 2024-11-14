from torch.utils.data import DataLoader
from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pl_bolts.optimizers.lars import LARS
from mirage.models.base import PretrainModel


class Mirage(PretrainModel):
    task = 'pretrain'
    def __init__(self, text_encoder, image_encoder, train_dataset=None, validation_dataset=None, test_dataset=None, embed_dim=768, num_workers=16,  gpus=[0], exclude_bn_bias = False, max_epochs=20, warmup_epochs=2, batch_size=16, optim='adam', scheduler='linear_warmup_cosine_annealing', learning_rate=1e-4, learning_rate_start=1e-6, learning_rate_end=1e-5, weight_decay=1e-6, frozen_text_encoder=False, pcl_temperature=0.07, otc_temperature=0.07, rcl_temperature=0.07, queue_temperature=0.07, queue_size=65536, warmup_temp=0.001, load_from=None, lambda_rcl=0.1, lambda_otc=1.0, lambda_pcl=1.0, save_each_epoch=False):
        super().__init__(text_encoder, image_encoder, train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset, embed_dim=embed_dim, num_workers=num_workers, gpus=gpus, exclude_bn_bias=exclude_bn_bias, max_epochs=max_epochs, warmup_epochs=warmup_epochs, batch_size=batch_size, optim=optim, scheduler=scheduler, learning_rate=learning_rate, learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end, weight_decay=weight_decay, frozen_text_encoder=frozen_text_encoder, save_each_epoch=save_each_epoch)
        self.pcl_logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / pcl_temperature))
        self.rcl_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / rcl_temperature))
        self.queue_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / queue_temperature))
        self.otc_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / otc_temperature))
        self.image_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.queue_size = queue_size
        self.current_queue_size = 0
        self.warmup_temp = warmup_temp
        self.register_buffer("text_queue", torch.randn(self.queue_size, self.embed_dim))
        self.text_queue = F.normalize(self.text_queue, dim=1)
        self.register_buffer("text_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.last_batch_size = None
        self.lambda_rcl = lambda_rcl
        self.lambda_otc = lambda_otc
        self.lambda_pcl = lambda_pcl
        self.inititalize_parameters()
        if load_from is not None:
            self.load_pretraind_weights(load_from)
     

    def load_pretraind_weights(self, pre_trained_path):
        state_dict = torch.load(pre_trained_path, map_location='cpu')['state_dict']
        self.load_state_dict(state_dict, strict=False)
    
    
    def inititalize_parameters(self):
        nn.init.normal_(self.image_proj.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_proj.weight, std=self.text_width ** -0.5)


    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
    
    def val_dataloader(self) :
        if self.validation_dataset is not None:
            return DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
    def text_dataloader(self) :
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)


    def shared_step(self, batch, stage='train'):
        image = batch['image']
        text = batch['text']

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        norm_image_features = F.normalize(image_features, dim=-1)
        norm_text_features = F.normalize(text_features, dim=-1)

        image_score, idx_image, q_text_features = self.find_nn(norm_image_features, self.text_queue) # [B, D]
        q_norm_text_features =  F.normalize(q_text_features, dim=-1)
        # [B, 1, N] * [B, N, 1] -> [B, 1]
        text_score = torch.bmm(norm_text_features.unsqueeze(1), q_norm_text_features.unsqueeze(2)).squeeze().detach()#.clamp(min=1e-6)

        robust_clip_dict = self.clip_loss(norm_image_features, q_norm_text_features, self.rcl_logit_scale.exp(), sample_weights = 1 - text_score)

        pair_clip_dict = self.clip_loss(norm_image_features, norm_text_features, self.pcl_logits_scale.exp(), sample_weights = 1 + text_score) 
        if self.training:
            self.dequeue_and_enqueue(norm_text_features, self.text_queue, self.text_queue_ptr)

        otc_dict = self.otc_loss(image_features, text_features, q_norm_text_features, self.otc_logit_scale.exp())

        loss_dict = {}
        loss_dict[stage + '_loss'] = 0
        for k, v in pair_clip_dict.items():
            loss_dict[stage + '_pair_' + k] = v
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v  * self.lambda_pcl
        for k, v in robust_clip_dict.items():
            loss_dict[stage + '_robust_' + k] = v 
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v  * self.lambda_rcl
        for k, v in otc_dict.items():
            loss_dict[stage + '_otc_' + k] = v 
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v * self.lambda_otc
        return loss_dict
    


    def otc_loss(self, image_embed, text_embed, q_text_embed, logit_scale, sample_weights=None):
        local_batch_size = image_embed.size(0)
        if local_batch_size != self.last_batch_size:
            self.clip_labels = local_batch_size * self.local_rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_batch_size = local_batch_size
        
        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        q_text_embed = F.normalize(q_text_embed, dim=-1, p=2)
        # gather features from all GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_embed_all= SyncFunction.apply(image_embed)
            text_embed_all= SyncFunction.apply(text_embed)
            q_text_embed_all = SyncFunction.apply(q_text_embed)
            
        else:
            image_embed_all= image_embed
            text_embed_all= text_embed
            q_text_embed_all = q_text_embed

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t() # [B, N*B]
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

       
        text_sim = text_embed @ q_text_embed_all.t()
        text_cost = - text_sim.detach() 


        image_loss = F.softmax(logits_per_image, dim=-1) * text_cost 
        image_loss = image_loss.sum(dim=-1) 
        if sample_weights is not None:
            image_loss = (image_loss * sample_weights).mean()
        else:  
            image_loss = image_loss.mean()

        
        text_loss = F.softmax(logits_per_text, dim=-1) * text_cost
        text_loss = text_loss.sum(dim=-1).mean() 


        loss = (image_loss + text_loss) / 2 

        return {'loss': loss}


    def encode_image(self, image):
        return self.image_proj(self.image_encoder(image))

    def encode_text(self, text):
        return self.text_proj(self.text_encoder(text))
    
    def training_step(self, batch, batch_idx):
        # step wise from 0 to 1 according to self.warmup_steps and global_step
        #self.warmup_temp = self.warmup_temp + (1 - self.warmup_temp) * self.global_step / (self.warmup_steps) + 1e-6
        #self.warmup_temp = min(1, self.warmup_temp)
        return super().training_step(batch, batch_idx)

    def on_train_start(self):
        # iteate the training dataloader to fill the queue
        torch.set_grad_enabled(False)
        for batch in self.train_dataloader():
            #image = batch['image']
            text = batch['text']
            for key in text.keys():
                text[key] = text[key].to(self.device)
            #image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            #norm_image_features = F.normalize(image_features, dim=-1)
            norm_text_features = F.normalize(text_features, dim=-1)
            b = text_features.size(0) * len(self.gpus)
            if self.text_queue_ptr + b >= self.queue_size:
                self.dequeue_and_enqueue(norm_text_features, self.text_queue, self.text_queue_ptr)
                break
            self.dequeue_and_enqueue(norm_text_features, self.text_queue, self.text_queue_ptr)

        torch.set_grad_enabled(True)
        return super().on_train_start()
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor, queue, queue_ptr):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features
        """
        #z = F.normalize(z, dim=1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            z = SyncFunction.apply(z)

        batch_size = z.shape[0]

        ptr = int(queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        queue[ptr : ptr + batch_size, :] = z
        #self.queue_y[ptr : ptr + batch_size] = y  # type: ignore
        ptr = (ptr + batch_size) % self.queue_size

        queue_ptr[0] = ptr  # type: ignore
    

    
    def find_nn(self, z: torch.Tensor, queue):
        """Finds the nearest neighbor of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """
        score = self.queue_logit_scale.exp() * (z @ queue.T) 
        score = F.softmax(score, dim=-1) # [B, N]
        idx = torch.argmax(score, dim=-1)
        # assign score on queue
        # [B, N] @ [N, D]
        nn = score @ queue
         
        return score, idx, nn
    

    def clip_loss(self, image_embed, text_embed, logit_scale, sample_weights=None, dig_scores=None):
        local_batch_size = image_embed.size(0)
        if local_batch_size != self.last_batch_size:
            self.clip_labels = local_batch_size * self.local_rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_batch_size = local_batch_size
    
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        # gather features from all GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_embed_all= SyncFunction.apply(image_embed)
            text_embed_all= SyncFunction.apply(text_embed)
            # if dig_scores is not None:
            #     dig_scores_all = SyncFunction.apply(dig_scores)
        else:
            image_embed_all= image_embed
            text_embed_all= text_embed
        #     if dig_scores is not None:
        #         dig_scores_all = dig_scores
        # # cosine similarity as logits
          
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        if dig_scores is not None:
            dig_logits_per_image = (logit_scale * image_embed @ dig_scores.t()).diag()
            dig_logits_per_text = (logit_scale * dig_scores @ image_embed.t()).diag()
     
            #print(dig_logits_per_image.shape, dig_logits_per_text.shape)
            # replace the logits
            logits_per_image[range(local_batch_size), self.clip_labels] = dig_logits_per_image
            logits_per_text[range(local_batch_size), self.clip_labels] = dig_logits_per_text
      
        # with torch.no_grad():
        #     inter =  text_embed @ text_embed_all.t()
        #     print(inter)
        if sample_weights is None:
            image_loss = F.cross_entropy(logits_per_image, self.clip_labels)
            text_loss = F.cross_entropy(logits_per_text, self.clip_labels)
        else:
            image_loss = F.cross_entropy(logits_per_image, self.clip_labels, reduction='none')
            text_loss = F.cross_entropy(logits_per_text, self.clip_labels, reduction='none')
            # weighted [B, 1] on sample wise
            image_loss = (image_loss * sample_weights).mean()
            text_loss = (text_loss * sample_weights).mean()
        loss = (image_loss + text_loss) / 2
        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.clip_labels).sum()
            acc = 100 * correct / local_batch_size
        return {'clip_loss': loss, 'clip_acc': acc}

    def forward(self, image=None, text_tokens=None, ori_text_tokens=None):
        outputs = {}
        if image is not None:
            image_features = self.encode_image(image)
            outputs['image_features'] = image_features
        if text_tokens is not None:
            text_features = self.encode_text(text_tokens)
            outputs['text_features'] = text_features
        if ori_text_tokens is not None:
            ori_text_features = self.encode_text(ori_text_tokens)
            outputs['ori_text_features'] = ori_text_features
        return outputs