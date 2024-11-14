import PIL.Image as Image
import os
import sys
import logging
log = logging.getLogger(__name__)
sys.path.append(os.getcwd())
from mirage.data.pretrain.base import PretrainDataset
from transformers import AutoTokenizer
import json
import pandas as pd
from torchvision import transforms
import torch
import random
from PIL import Image
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PmcOaDataset(PretrainDataset):

    def __init__(self, root_path, dataset_path, num_colors=3, image_transform=[], text_transform=[], rate=1.0, max_length=77, pretrained_name='microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract', mask_rate=0, fp_rate=0, fn_rate=0, noise_rate=0, aug_img=False):
        super().__init__(dataset_path, image_transform, text_transform, rate)
        self.root_path = root_path
        self.max_length = max_length
        self.rate = rate
        self.mask_rate = mask_rate
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.noise_rate = noise_rate
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.aug_img = aug_img

        if aug_img:
            self.aug_transform = self._build_aug()
        



    def _load_dataset(self):
        self.data = pd.read_json(self.dataset_path, lines=True)
    
    def _load_statics(self):
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

    def _build_transform(self):
        self.image_transform += [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        self.image_transform = transforms.Compose(self.image_transform)

    def _build_aug(self):
        # get size frp, the image transform Compose
        
        size = self.image_transform.transforms[0].size
        return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])


    def _tokenize(self, text):
        tokens = self.tokenizer(
        text,
        max_length=self.max_length,
        add_special_tokens=True,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
        for key, token in tokens.items():
            tokens[key] = token.squeeze(dim=0)
        return tokens
    
    def _mask_tokens(self, tokens):
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        mask =  torch.rand(input_ids.shape) < self.mask_rate
        mask[~attention_mask.bool()] = False
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.tokenizer.mask_token_id
        masked_tokens = {'input_ids': masked_input_ids, 'attention_mask': attention_mask}
        return masked_tokens
        
    def _get_image(self, index):
        image = Image.open(os.path.join(self.root_path, self.data['image'][index])).convert('RGB')
        if self.aug_img:
            return self.aug_transform(image), self.image_transform(image)
        return self.image_transform(image)
    
    def _get_text(self, index):
        return self.data['caption'][index]
    
    def __getitem__(self, index):
        return_dict = {}
        if self.aug_img:
            aug_image, image = self._get_image(index)
            return_dict['aug_image'] = aug_image
        else:
            image = self._get_image(index)
        if self.fp_rate > 0:
            rand = random.random()
            if rand < self.fp_rate:
                index = random.randint(0, self.__len__()-1)
        text = self._get_text(index)
        if self.tokenizer is not None:
            text = self._tokenize(text)
        if self.mask_rate > 0:
            masked_text = self._mask_tokens(text)
            #return {'image': image, 'text': text, 'masked_text': masked_text}
            return_dict['masked_text'] = masked_text
        return_dict['image'] = image
        return_dict['text'] = text
        return return_dict
    
    def __len__(self):
        return int(self.rate * len(self.data))
    

    






        