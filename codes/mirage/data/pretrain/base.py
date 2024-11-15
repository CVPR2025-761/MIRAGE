
from torch.utils.data import Dataset
import logging
from torch.utils.data import Dataset



log = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    def __init__(self, dataset_path, image_transform, text_transform, rate):
        self.dataset_path = dataset_path
        self._load_dataset()
        self._load_statics()

        self.image_transform = image_transform.copy()
        self.text_transform = text_transform.copy()
        self.rate = rate
        self._build_transform()

    def _build_transform(self):
        self.image_transform += [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        self.image_transform = transforms.Compose(self.image_transform)

    def _load_dataset(self):
        raise NotImplementedError

    def _load_statics(self):
        raise NotImplementedError
    
    def _get_image(self, index):
        raise NotImplementedError

    def _get_text(self, index):
        raise NotImplementedError

    def _tokenize(self, text):
        raise  NotImplementedError
    
    def __getitem__(self, index):
        image = self._get_image(index)
        text = self._get_text(index)
        if self.tokenizer is not None:
            text = self._tokenize(text)
        return {'image': image, 'text': text}

    def __len__(self):
        raise NotImplementedError


