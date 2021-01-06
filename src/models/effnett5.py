import copy
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from src.metrics import compute_exact, compute_f1
from torchvision import transforms
from transformers import T5ForConditionalGeneration, T5Tokenizer

EFFNET_OUTPUT_SIZE = {
    'efficientnet-b0': 1280,
    'efficientnet-b1': 1280,
    'efficientnet-b2': 1408,
    'efficientnet-b3': 1536,
    'efficientnet-b4': 1792,
    'efficientnet-b5': 2048,
    'efficientnet-b6': 2304,
    'efficientnet-b7': 2560,
}


class LitEffNetT5(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.use_t5_only and not self.hparams.use_image_ocr:
            raise ValueError(
                'You must set `use_image_ocr` to True if `use_t5_only` is True.'
            )

        self.effnet = EfficientNet.from_pretrained(
            self.hparams.effnet_model,
            advprop=self.hparams.use_advprop
        )

        if self.hparams.freeze_effnet:
            print('Freezing EfficientNet...')
            for param in self.effnet.parameters():
                param.requires_grad = False

        self.t5 = T5ForConditionalGeneration.from_pretrained(
            self.hparams.t5_model
        )

        # The T5 implementation uses the same Embedding module for both the
        # encoder and the decoder blocks. See
        # https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_t5.html#T5ForConditionalGeneration
        # We want to compute our own embeddings to feed the encoder but keep
        # the original one for the decoder, so we must separate them into
        # independent modules. To do this we will copy the shared embedding
        # module and use it to redefine the encoders module.
        # NOTE: the t5 model should always be called with `inputs_embeds`, not
        # with `input_ids`.
        shared_embeddings = self.t5.get_input_embeddings()
        encoder_embeddings = copy.deepcopy(shared_embeddings)
        self.t5.encoder.set_input_embeddings(encoder_embeddings)

        if self.hparams.freeze_t5:
            print('Freezing T5...')
            for param in self.t5.parameters():
                param.requires_grad = False

        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_model)

        # The bounding box for each token should contain the coordinates
        # of the corners corresponding to the boxes comprising the word to
        # which it is part, thus the input tensor is expected to have the
        # dimensions (B, S, 8) and the output must match the model embedding
        # dimension, i.e. (B, S, D), where B -> batch, S -> sequence, D -> model
        self.bbox_embedding = nn.Linear(8, self.t5.config.d_model)

        # The image embedding layer takes the features extracted by the effnet
        # and converts them to the appropriate dimensions to match the model
        # embedding
        self.image_embedding = nn.Conv2d(
            in_channels=EFFNET_OUTPUT_SIZE[self.hparams.effnet_model],
            out_channels=self.t5.config.d_model,
            kernel_size=1
        )

        self.collate_fn = None
        self.image_transforms = transforms.Compose([
            # transforms.Resize(size=(740, 560)),
            transforms.Resize(size=(640, 480)),
            transforms.ToTensor()
        ])

        # self.count = 0

    def generate(self, inputs_embeds):

        decoder_input_ids = torch.full(
            size=(inputs_embeds.shape[0], 1),
            fill_value=self.t5.config.decoder_start_token_id,
            dtype=torch.long
        ).to(inputs_embeds.device)

        for step in range(self.hparams.target_max_len):

            logits = self.t5(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            ).logits

            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            if torch.eq(next_token_id[:, -1], self.tokenizer.eos_token_id).all():
                break

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token_id], dim=-1)

        return decoder_input_ids

    def _t5_only_forward(self, input_ids, input_mask, target_ids):

        if self.training:
            return self.t5(input_ids=input_ids,
                           attention_mask=input_mask,
                           labels=target_ids,
                           return_dict=True).loss

        else:
            return self.t5.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=self.hparams.target_max_len,
            )

    def forward(self, batch):

        img, img_bbox, input_ids, input_mask, input_bbox, _, \
            target_ids, _, _ = batch

        # print([self.tokenizer.decode(i) for i in input_ids[0]])

        if self.hparams.use_t5_only:
            self._t5_only_forward(input_ids, input_mask, target_ids)

        # compute token embeds
        # (B, S, 1) -> (B, S, D)
        inputs_embeds = self.t5.encoder.get_input_embeddings()(input_ids)

        # compute image embeds using the EfficientNet
        # from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        if self.hparams.use_image_features:
            # (B, C, H, W) -> (B, F, h, w)
            img_features = self.effnet.extract_features(img)
            # (B, F, h, w) -> (B, D, h, w)
            img_embeds = self.image_embedding(img_features)
            # (B, D, h, w) -> (B, h+w, D)
            img_embeds = img_embeds.permute(0, 2, 3, 1).reshape(
                img_embeds.shape[0], -1, self.t5.config.d_model
            )
            # concat embeds along the sequence axis
            # (B, S, D), (B, h+w, D) -> (B, S+h+w, D)
            inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)

        # compute ocr bbox embeds
        if self.hparams.use_ocr_bbox:
            if self.hparams.use_image_features:
                # (B, 1, 8) -> (B, h+w, 8)
                img_bbox_expanded = img_bbox.expand(-1,
                                                    img_embeds.shape[1], -1)
                # (B, S, 8), (B, h+w, 8) -> (B, S+h+w, 8)
                bbox = torch.cat([input_bbox, img_bbox_expanded], dim=1)
                # (B, S+h+w, 8) -> (B, S+h+w, D)
                bbox_embeds = self.bbox_embedding(bbox.float())
                # adds token and bbox embeds
                inputs_embeds += bbox_embeds
            else:
                inputs_embeds += self.bbox_embedding(input_bbox.float())

        if self.training:
            return self.t5(inputs_embeds=inputs_embeds,
                           labels=target_ids,
                           return_dict=True).loss

        else:
            return self.generate(inputs_embeds)

    def training_step(self, batch, batch_idx):

        loss = self(batch)

        try:
            self.logger.experiment.log_metric('train_loss', loss)
        except AttributeError:
            pass

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        img, img_bbox, input_ids, input_mask, input_bbox, question, \
            target_ids, target_mask, target_text = batch

        predicted_ids = self(batch)
        preds = [self.tokenizer.decode(ids) for ids in predicted_ids]

        batch_size = img.shape[0]
        exact = sum([compute_exact(ans, pred)
                     for ans, pred in zip(target_text, preds)]) / batch_size
        f1 = sum([compute_f1(ans, pred)
                  for ans, pred in zip(target_text, preds)]) / batch_size

        # if self.count % 10 == 0:
        #     print('\nVALIDATION EXAMPLE CHECK ' + 55 * '-')
        #     fig, ax = plt.subplots(figsize=(8, 8))
        #     ax.imshow(img[0].to('cpu').numpy().transpose(1, 2, 0))
        #     ax.axis('off')
        #     plt.show()
        #     print('Question..:', question[0])
        #     print('Answer....:', target_text[0])
        #     print('Prediction:', preds[0])
        # self.count += 1

        return {'val_exact': exact, 'val_f1': f1}

    def test_step(self, batch, batch_idx):

        img, img_bbox, input_ids, input_mask, input_bbox, question, \
            target_ids, target_mask, target_text = batch

        predicted_ids = self(batch)
        preds = [self.tokenizer.decode(ids) for ids in predicted_ids]

        batch_size = img.shape[0]
        exact = sum([compute_exact(ans, pred)
                     for ans, pred in zip(target_text, preds)]) / batch_size
        f1 = sum([compute_f1(ans, pred)
                  for ans, pred in zip(target_text, preds)]) / batch_size

        # if self.count % 10 == 0:
        #     print('\nTEST EXAMPLE CHECK ' + 55 * '-')
        #     fig, ax = plt.subplots(figsize=(8, 8))
        #     ax.imshow(img[0].to('cpu').numpy().transpose(1, 2, 0))
        #     ax.axis('off')
        #     plt.show()
        #     print('Question..:', question[0])
        #     print('Answer....:', target_text[0])
        #     print('Prediction:', preds[0])
        # self.count += 1

        return {'test_exact': exact, 'test_f1': f1}

    def validation_epoch_end(self, outputs):

        avg_exact = sum([x['val_exact'] for x in outputs]) / len(outputs)
        avg_f1 = sum([x['val_f1'] for x in outputs]) / len(outputs)

        try:
            self.logger.experiment.log_metrics({
                'avg_val_exact': avg_exact,
                'avg_val_f1': avg_f1
            })
        except AttributeError:
            pass

        self.log('avg_val_exact', avg_exact)
        self.log('avg_val_f1', avg_f1)

    def test_epoch_end(self, outputs):

        avg_exact = sum([x['test_exact'] for x in outputs]) / len(outputs)
        avg_f1 = sum([x['test_f1'] for x in outputs]) / len(outputs)

        try:
            self.logger.experiment.log_metrics({
                'avg_val_exact': avg_exact,
                'avg_val_f1': avg_f1
            })
        except AttributeError:
            pass

        self.log('avg_val_exact', avg_exact)
        self.log('avg_val_f1', avg_f1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate, eps=1e-08)

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--effnet_model', type=str, required=True)
        parser.add_argument('--use_advprop', action='store_true')
        parser.add_argument('--use_image_features', action='store_true')
        parser.add_argument('--use_ocr_bbox', action='store_true')
        parser.add_argument('--use_synthetic_images', action='store_true')
        parser.add_argument('--freeze_effnet', action='store_true')
        parser.add_argument('--t5_model', type=str, required=True)
        parser.add_argument('--use_t5_only', action='store_true')
        parser.add_argument('--freeze_t5', action='store_true')

        return parser
