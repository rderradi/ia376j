from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
from torchvision import transforms
from transformers import T5Tokenizer

from .dataset import DocVQADataset


class DocVQADataModule(pl.LightningDataModule):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.hparams.image_height, self.hparams.image_width)
            ),
            transforms.ToTensor()
        ])

        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_model)

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            self.train = DocVQADataset(
                mode='train',
                data_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                use_image_ocr=self.hparams.use_image_ocr,
                use_synthetic_images=self.hparams.use_synthetic_images,
                input_max_len=self.hparams.input_max_len,
                target_max_len=self.hparams.target_max_len,
                transform=self.transform,
            )
            self.val = DocVQADataset(
                mode='val',
                data_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                use_image_ocr=self.hparams.use_image_ocr,
                use_synthetic_images=self.hparams.use_synthetic_images,
                input_max_len=self.hparams.input_max_len,
                target_max_len=self.hparams.target_max_len,
                transform=self.transform,
            )

            self.dims = tuple(self.train[0][0].shape)

        if stage == 'test' or stage is None:

            self.test = DocVQADataset(
                mode='val',
                data_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                use_image_ocr=self.hparams.use_image_ocr,
                use_synthetic_images=self.hparams.use_synthetic_images,
                input_max_len=self.hparams.input_max_len,
                target_max_len=self.hparams.target_max_len,
                transform=self.transform,
            )

            self.dims = tuple(self.test[0][0].shape)

    def train_dataloader(self):
        return self.train.get_dataloader(
            batch_size=self.hparams.batch_size,
            shuffle=True,
            # num_workers=0,
            # collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return self.val.get_dataloader(
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=0,
            # collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return self.test.get_dataloader(
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # num_workers=0,
            # collate_fn=self.collate_fn
        )

    @staticmethod
    def add_module_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--image_height', type=int, default=640)
        parser.add_argument('--image_width', type=int, default=480)
        parser.add_argument('--input_max_len', type=int, default=512)
        parser.add_argument('--target_max_len', type=int, default=128)
        parser.add_argument('--use_image_ocr', action='store_true')

        return parser
