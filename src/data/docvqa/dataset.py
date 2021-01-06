import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer


class DocVQADataset(Dataset):
    """Dataset class to handle DocVQA dataset.

    Args:
        mode: either `train`, `val` or `test`
        tokenizer: a T5 tokenizer instance
        input_max_len: maximum lenght of the input sequence
        target_max_len: maximum lenght of the target sequence
        transform: image transforms to be applied
        seed: random state seed
    """

    DATASET_FILE = {
        'train': Path('train_v1.0.json'),
        'val': Path('val_v1.0.json'),
        'test': Path('test_v1.0.json')

    }

    def __init__(self,
                 mode: str,
                 data_dir: str,
                 use_image_ocr: bool,
                 use_synthetic_images: bool,
                 input_max_len: int,
                 target_max_len: int,
                 tokenizer: T5Tokenizer = None,
                 transform: List[Callable] = None):
        super().__init__()

        assert mode in ['train', 'val', 'test'], ('mode must be either '
                                                  'train, val or test')

        self.data_dir = Path(data_dir) / Path(mode)
        with open(self.data_dir / self.DATASET_FILE[mode], 'r') as f:
            dataset = json.load(f)

        assert dataset['dataset_split'] == mode

        self.data = dataset['data']

        self.tokenizer = tokenizer
        self.use_image_ocr = use_image_ocr
        self.use_synthetic_images = use_synthetic_images
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def _get_image(self, idx):

        img_file = Path(self.data[idx]['image'])
        image = Image.open(self.data_dir / img_file).convert('RGB')

        width, height = image.size
        image_bbox = torch.tensor([[0, 0, width, 0, width, height, 0, height]])

        return image, image_bbox

    def _get_ocr_text(self, idx):

        ocr_file = Path(self.data[idx]['image']
                        .replace('documents', 'ocr_results')
                        .replace('.png', '.json'))

        with open(self.data_dir / ocr_file, 'r') as f:
            ocr = json.load(f)

        lines = ocr['recognitionResults'][0]['lines']
        text = ' '.join([w['text'] for l in lines for w in l['words']])

        return text

    def _generate_image(self, idx):

        ocr_file = Path(self.data[idx]['image']
                        .replace('documents', 'ocr_results')
                        .replace('.png', '.json'))

        with open(self.data_dir / ocr_file, 'r') as f:
            ocr = json.load(f)

        height = ocr['recognitionResults'][0]['height']
        width = ocr['recognitionResults'][0]['width']

        lines = ocr['recognitionResults'][0]['lines']
        words = [w for l in lines for w in l['words']]

        image = Image.new('RGB', (width, height), (255, 255, 255))
        d = ImageDraw.Draw(image)

        for word in words:
            text = word['text']
            bbox = word['boundingBox']

            # left-bottom
            x0, y0 = bbox[0:2]

            # right-top
            x1, y1 = bbox[4:6]

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)

            font_size = int(min(dx, dy) * 0.8)

            if font_size <= 0:
                continue

            # for debug
            # d.rectangle(((x0, y0), (x1, y1)), fill='yellow')

            font = ImageFont.truetype('arial.ttf', font_size)
            d.text((x0, y0), text=text, font=font, fill=(0, 0, 0))

        image_bbox = torch.tensor([[0, 0, width, 0, width, height, 0, height]])

        return image, image_bbox

    def _get_ocr(self, idx):

        ocr_file = Path(self.data[idx]['image']
                        .replace('documents', 'ocr_results')
                        .replace('.png', '.json'))

        with open(self.data_dir / ocr_file, 'r') as f:
            ocr = json.load(f)

        lines = ocr['recognitionResults'][0]['lines']
        words = [{'boundingBox': [0] * 8, 'text': '[ocr]'}]
        words.extend([w for l in lines for w in l['words']])

        ocr_text = []
        ocr_ids = []
        ocr_bbox = []
        for word in words:
            text = word['text']
            ids = self.tokenizer(text=text)['input_ids'][:-1]
            bbox = [word['boundingBox']] * len(ids)
            ocr_text.append(text)
            ocr_ids.extend(ids)
            ocr_bbox.extend(bbox)

        assert len(ocr_ids) == len(ocr_bbox)

        ocr_text = ' '.join(ocr_text)

        return ocr_text, ocr_ids, ocr_bbox

    def _get_question(self, idx):

        text = self.data[idx]['question']
        ids = self.tokenizer(text=text)['input_ids'][:-1]
        bbox = [[0] * 8] * len(ids)

        return text, ids, bbox

    def _get_answer(self, idx):

        text = random.choice(self.data[idx]['answers'])
        tokenization = self.tokenizer(
            text=text,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids = tokenization['input_ids'].squeeze(0)
        mask = tokenization['attention_mask'].squeeze(0)

        return text, ids, mask

    def _prepare_inputs(self, question_ids, question_bbox, ocr_ids, ocr_bbox):

        input_ids = question_ids + ocr_ids
        input_mask = [1] * len(input_ids)
        input_bbox = question_bbox + ocr_bbox

        padding = max(0, self.input_max_len - len(input_ids))

        input_ids = input_ids[:self.input_max_len] + [0] * padding
        input_mask = input_mask[:self.input_max_len] + [0] * padding
        input_bbox = input_bbox[:self.input_max_len] + [[0] * 8] * padding

        assert len(input_ids) == len(input_mask) == len(
            input_bbox) == self.input_max_len

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_bbox = torch.tensor(input_bbox)

        return input_ids, input_mask, input_bbox

    def __getitem__(self, idx):

        if self.use_synthetic_images:
            img, img_bbox = self._generate_image(idx)
        else:
            img, img_bbox = self._get_image(idx)

        if self.transform:
            img = self.transform(img)

        question, question_ids, question_bbox = self._get_question(idx)
        ocr_text, ocr_ids, ocr_bbox = self._get_ocr(idx)

        input_ids, input_mask, input_bbox = self._prepare_inputs(
            question_ids, question_bbox, ocr_ids, ocr_bbox
        )

        answer, answer_ids, answer_mask = self._get_answer(idx)

        return img, img_bbox, input_ids, input_mask, input_bbox, question, \
            answer_ids, answer_mask, answer

    def inspect_example(self, idx):
        """ Inspect an example image with question and answers.

        Args:
            idx: the index of the example to be plotted
        """

        img_tensor, img_bbox, input_ids, input_mask, input_bbox, question, \
            answer_ids, answer_mask, answer = self[idx]

        orig_image = self._get_image(idx)
        ocr_image = self._generate_image(idx)
        ocr_text = self._get_ocr_text(idx)

        try:
            fig, ax = plt.subplots(1, 2, figsize=(25, 25))
            ax[0].imshow(orig_image)
            ax[0].axis('off')
            ax[1].imshow(ocr_image)
            ax[1].axis('off')
            fig.tight_layout()
            plt.show()
        except TypeError:
            pass

        print('--> OCR text:', ocr_text)
        print('--> question text:', question)
        print('--> answer text:', answer)
        print('--> image:', img_tensor)
        print('--> image bbox:', img_bbox)
        print('--> input token ids:', input_ids)
        print('--> input mask:', input_mask)
        print('--> input bbox:', input_bbox)
        print('--> answer token ids:', answer_ids)
        print('--> answer mask:', answer_mask)

    def get_dataloader(self, batch_size: int, shuffle: bool = False,
                       collate_fn: Callable = None,
                       num_workers: int = mp.cpu_count()) -> DataLoader:
        """ Returns a DataLoader for this Dataset.

        Args:
            batch_size: number of samples per batch to load
            shuffle: whether to shuffle the data at every epoch
            num_workers: how many subprocesses to use for data loading

        Returns:
            A DataLoader instance for this dataset.
        """
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          pin_memory=True,
                          collate_fn=collate_fn,
                          num_workers=num_workers)
