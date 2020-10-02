# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.data import data_utils

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    base_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.hub_utils import GeneratorHubInterface
from .transformer_mass import mass_base, mass_tiny, TransformerMASSModel
from ..modules.output_layer import ClassificationHead, MLMHead, SequenceTaggingHead

HOME_PATH = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
logger = logging.getLogger(__name__)


@register_model('transformer_multitask')
class TransformerMultitaskModel(TransformerMASSModel):

    @classmethod
    def hub_models(cls):

        # fmt: off
        def ner_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'checkpoint_file': 'checkpoint72.17.pt',
            }

        def sum_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.50.pt',
            }

        def sum_as_lm_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.50.pt',
                'arch': 'transformer_lm_base',
            }

        def pretrain_config(path):
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.pt',
            }

        def pretrain_lm_config(path):
            """for language model tasks"""
            return {
                'path': path,
                'tokenizer': None,
                'bpe': 'bert',
                'bpe_vocab_file': os.path.join(path, 'dict.txt'),
                'no_repeat_ngram_size': 3,
                'min_len': 50,
                'checkpoint_file': 'checkpoint72.pt',
                'arch': 'transformer_lm_base',
            }

        def cls_config(path):
            pass

        return {
            'transformer.pretrain': pretrain_config(HOME_PATH + '/models/pretrain/'),
            'transformer.pretrain.lm': pretrain_lm_config(HOME_PATH + '/models/pretrain/'),
            'transformer.ft.sum.lm': sum_as_lm_config(HOME_PATH + '/models/finetune/sum/'),
            'transformer.ft.sum': sum_config(HOME_PATH + '/models/finetune/sum/'),
            'transformer.ft.tag': ner_config(HOME_PATH + '/models/finetune/ner/'),
            'transformer.ft.cls': cls_config(HOME_PATH + '/models/finetune/cls/'),
        }

    @staticmethod
    def add_args(parser):
        TransformerMASSModel.add_args(parser)
        parser.add_argument('--share-encoder-input-output-embed', action='store_true',
                            help='')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='checkpoint.pt', data_name_or_path='.', bpe='bert',
                        **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return PretrainHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = super().build_model(args, task)
        model.mlm_head = MLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(model.encoder.dictionary),
            activation_fn=args.activation_fn,
            weight=model.encoder.embed_tokens.weight if args.share_encoder_input_output_embed else None,
        )
        # or define in __init__ function
        model.classification_heads = nn.ModuleDict()
        model.tagging_heads = nn.ModuleDict()
        return model

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a sentence classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_tagging_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a sequence tagging head."""
        if name in self.tagging_heads:
            prev_num_classes = self.tagging_heads[name].out_proj.out_features
            prev_inner_dim = self.tagging_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.tagging_heads[name] = SequenceTaggingHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            use_crf='crf' in name
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def get_mlm_output(self, encoder_out, masked_tokens=None):
        """Project features to the vocabulary size."""
        return self.mlm_head(encoder_out, masked_tokens)

    def get_clm_output(self, decoder_out, masked_tokens=None):
        """sentence generation head, used for translation, summarization."""
        return decoder_out[masked_tokens]

    def get_cls_output(self, encoder_out, masked_tokens=None, classification_head_name=None):
        """
        We "pool" the model by simply taking the hidden state corresponding to the first token.
        This is necessary for sentence-level classification tasks.
        input: [batch_size, seq_length, hidden_size]
        output: [batch_size, hidden_size]
        """
        if masked_tokens is not None:
            features = encoder_out[masked_tokens, 0, :]
        else:
            features = encoder_out[:, 0, :]
        return self.classification_heads[classification_head_name](features)

    def get_tag_output(self, encoder_out, masked_tokens, tagging_head_name):
        return self.tagging_heads[tagging_head_name].decode(encoder_out, masked_tokens)

    def get_tag_loss(self, encoder_out, masked_tokens, tagging_head_name, tags):
        return self.tagging_heads[tagging_head_name](encoder_out, masked_tokens, tags)

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, clm_positions=None,
                masked_tokens=None, features_only=False, classification_head_name=None, tagging_head_name=None,
                tags=None):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): `(batch)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
        """
        if classification_head_name is not None and 'pretrain' not in classification_head_name:
            features_only = True

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        if torch.isnan(encoder_out.encoder_out).any():
            print('catch encoder non')

        encoder_feature = encoder_out.encoder_out.transpose(0, 1)  # T x B x C -> B x T x C

        # 1. encoder model
        if features_only:
            return encoder_feature

        # 2. encoder-decoder model
        decoder_out = None
        extra = {}
        if prev_output_tokens is not None and prev_output_tokens.numel() > 0:
            decoder_out, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, positions=clm_positions)
            if torch.isnan(decoder_out).any():
                print('catch decoder nan')

        if masked_tokens:
            if masked_tokens.get('clm', None) is not None:
                extra['clm_out'] = self.get_clm_output(decoder_out, masked_tokens['clm'])
            if masked_tokens.get('mlm', None) is not None:
                extra['mlm_out'] = self.get_mlm_output(encoder_feature, masked_tokens['mlm'])
            if masked_tokens.get('cls', None) is not None:
                extra['cls_out'] = self.get_cls_output(encoder_feature, masked_tokens['cls'],
                                                       classification_head_name)  # masked_tokens['cls']是干嘛的？
            if masked_tokens.get('tag', None) is not None:
                extra['tag_out'] = self.get_tag_loss(encoder_feature, masked_tokens['tag'], tagging_head_name, tags)

        return decoder_out, extra

    def get_mlm_targets(self, sample, net_output):
        return sample["mlm_target"]

    def get_clm_targets(self, sample, net_output):
        return sample["clm_target"]

    def get_cls_targets(self, sample, net_output):
        return sample["cls_target"]

    def get_tag_targets(self, sample, net_output):
        return sample["tag_target"]

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + '.' if name != '' else ''
        keys_to_delete = []

        # Handle new classification heads present in the state dict.
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)

            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

        # Handle new tagging heads present in the state dict.
        current_tagging_head_names = [] if not hasattr(self, 'tagging_heads') else \
            list(self.tagging_heads.keys())
        for k in state_dict.keys():
            if not k.startswith(prefix + 'tagging_heads.'):
                continue

            head_name = k[len(prefix + 'tagging_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'tagging_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'tagging_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_tagging_head_names:
                    self.register_tagging_head(head_name, num_classes, inner_dim)

            else:
                if head_name not in current_tagging_head_names:
                    logger.warning(
                        'deleting tagging head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.tagging_heads[head_name].out_proj.out_features
                        or inner_dim != self.tagging_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting tagging head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

        # delete ner_head in old version. git checkout aa117c57cf710309672c808b9973ea35633bf796
        for k in state_dict.keys():
            if k.startswith('ner_head.'):
                keys_to_delete.append(k)

        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            truncate_emb('encoder.embed_tokens.weight')
            truncate_emb('decoder.embed_tokens.weight')
            truncate_emb('encoder.output_projection.weight')
            truncate_emb('decoder.output_projection.weight')

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

        if hasattr(self, 'tagging_heads'):
            cur_state = self.tagging_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'tagging_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'tagging_heads.' + k)
                    state_dict[prefix + 'tagging_heads.' + k] = v

        # Copy mlm_head into state_dict
        cur_state = self.mlm_head.state_dict()
        for k, v in cur_state.items():
            k_str = prefix + 'mlm_head.' + k
            if k_str not in state_dict:
                logger.info('add ' + k_str + ' to loaded state_dict')
                state_dict[k_str] = v


class PretrainHubInterface(GeneratorHubInterface):

    def __init__(self, args, task, models):
        super().__init__(args, task, models)
        self.model = models[0]
        self.masked_token = '[MASK]'

    def encode_masked_input(self, masked_input):
        text_spans = masked_input.split(self.masked_token)
        text_spans_bpe = (' {0} '.format(self.masked_token)).join(
            [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
        ).strip()
        tokens = self.task.source_dictionary.encode_line(
            '[CLS] ' + text_spans_bpe + ' [SEP]',
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens

    def fill_single_mask(self, masked_inputs, topk=3):
        if isinstance(masked_inputs, str):
            masked_inputs = [masked_inputs]
        assert all(self.masked_token in masked_input for masked_input in masked_inputs), \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(self.masked_token)

        tokens = [self.encode_masked_input(masked_input) for masked_input in masked_inputs]
        pad_to_length = max(len(token) for token in tokens)

        tokens = data_utils.collate_tokens(
            tokens,
            self.task.source_dictionary.pad(),
            self.task.source_dictionary.eos(),
            False, False,
            pad_to_length=pad_to_length,
        )
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        src_lengths = tokens.ne(self.task.source_dictionary.pad()).sum(dim=-1)
        masked_tokens = tokens.eq(self.task.source_dictionary.mask_index)
        # with utils.model_eval(self.model):  # new version
        with utils.eval(self.model):
            logits = self.model.forward_encoder(
                tokens.long().to(device=self.device),
                src_lengths=src_lengths.to(device=self.device),
                masked_tokens=masked_tokens
            )
        prob = logits.softmax(dim=-1)
        all_values, all_index = prob.topk(k=topk, dim=-1)
        topk_predicted_token_bpe = self.task.source_dictionary.string(all_index)

        topk_predicted_token_bpe = [tokens.split(' ') for tokens in topk_predicted_token_bpe.split('\n')]
        return topk_predicted_token_bpe

    def fill_multi_mask(self, masked_inputs, topk=3, return_filled_sentence=False):
        pass

    def fill_mask(self, masked_inputs, topk=3, return_filled_sentence=False):
        if isinstance(masked_inputs, str):
            masked_inputs = [masked_inputs]
        masked_token = '[MASK]'
        assert all(masked_token in masked_input for masked_input in masked_inputs), \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)

        def encode_masked_input(masked_input):
            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '[CLS] ' + text_spans_bpe + ' [SEP]',
                append_eos=False,
                add_if_not_exist=False,
            )
            return tokens

        tokens = [encode_masked_input(masked_input) for masked_input in masked_inputs]
        pad_to_length = max(len(token) for token in tokens)

        tokens = data_utils.collate_tokens(
            tokens,
            self.task.source_dictionary.pad(),
            self.task.source_dictionary.eos(),
            False, False,
            pad_to_length=pad_to_length,
        )
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        src_lengths = tokens.ne(self.task.source_dictionary.pad()).sum(dim=-1)
        masked_tokens = tokens.eq(self.task.source_dictionary.mask_index)
        # with utils.model_eval(self.model):  # new version
        with utils.eval(self.model):
            logits = self.model.forward_encoder(
                tokens.long().to(device=self.device),
                src_lengths=src_lengths.to(device=self.device),
                masked_tokens=masked_tokens
            )
        prob = logits.softmax(dim=-1)
        all_values, all_index = prob.topk(k=topk, dim=-1)
        topk_predicted_token_bpe = self.task.source_dictionary.string(all_index)

        topk_predicted_token_bpe = [tokens.split(' ') for tokens in topk_predicted_token_bpe.split('\n')]
        if not return_filled_sentence:
            return topk_predicted_token_bpe

        # all_outputs = []
        # topk_predicted_token_bpe = iter(topk_predicted_token_bpe)
        # topk_filled_outputs = []
        # for masked_input in masked_inputs:
        #         predicted_token = self.bpe.decode(predicted_token_bpe)
        #         if predicted_token_bpe.startswith('\u2581'):
        #             predicted_token = ' ' + predicted_token
        #         if " {0}".format(masked_token) in masked_input:
        #             topk_filled_outputs.append((
        #                 masked_input.replace(
        #                     ' {0}'.format(masked_token), predicted_token
        #                 ),
        #                 values[index].item(),
        #                 predicted_token,
        #             ))
        #         else:
        #             topk_filled_outputs.append((
        #                 masked_input.replace(masked_token, predicted_token),
        #                 values[index].item(),
        #                 predicted_token,
        #             ))
        #     all_outputs.append(topk_filled_outputs)
        return None

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """ Winograd Schema Challenge task (WSC)
        https://github.com/pytorch/fairseq/tree/master/examples/roberta#pronoun-disambiguation-winograd-schema-challenge
        """
        pass

    def register_classification_head(
            self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def register_tagging_head(
            self, name: str, num_tags: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_tagging_head(
            name, num_tags=num_tags, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def sequence_tagging(self, tokens: torch.LongTensor, head: str):
        tokens = tokens[:-1].view(-1, 1)
        src_lengths = torch.LongTensor([tokens.numel()]).view(-1, 1)
        r = self.models[0].decode(tokens, tagging_head_name=head, src_lengths=src_lengths)
        return r

    # CTC、text infilling
    # def punc_revise(self):
    #     pass

    # def constrain_decode(self):
    #     """
    #     https://github.com/pytorch/fairseq/blob/master/tests/test_constraints.py
    #     """
    #     pass


@register_model('transformer_pretrain_lm')
class TransformerLanguageModel(TransformerMultitaskModel):
    """ Auto regressive language model """

    def forward(self, src_tokens, **kwargs):
        decoder_out = self.decoder(src_tokens, **kwargs)
        return decoder_out

    def forward_encoder(self, src_tokens, src_lengths=None, masked_tokens=None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_feature = encoder_out.encoder_out.transpose(0, 1)
        mlm_out = self.get_mlm_output(encoder_feature, masked_tokens)
        return mlm_out


@register_model('transformer_mrc')
class TransformerMRCModel(TransformerMultitaskModel):
    """
    dataset：
      - zh：CMRC-2018 https://hfl-rc.github.io/cmrc2018/
      - en SQuAD、SQuADv2、 The Stanford Question Answering Dataset
        leaderboard：https://rajpurkar.github.io/SQuAD-explorer/
        <s> Passage here. </s> Q: Question here? </s>   https://github.com/ecchochan/roberta-squad
      - em NewsQA
      - em CommensenseQA
      - leaderboard：https://github.com/brightmart/roberta_zh#%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%B5%8B%E8%AF%95
    code：
      - https://github.com/CLUEbenchmark/CLUEPretrainedModels/tree/master/baselines/models_pytorch/mrc_pytorch
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/cmrc2018_finetune_pytorch.py
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/CJRC_finetune_pytorch.py
      - https://github.com/ewrfcas/bert_cn_finetune/blob/master/DRCD_finetune_pytorch.py
    """
    pass


@register_model('transformer_pretrain_tagging')
class TransformerTaggingModel(TransformerMultitaskModel):
    """Bidirectional Language Model for Sequence Tagging"""

    def forward(self, src_tokens, masked_tokens=None, tagging_head_name=None, tags=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        log_likelihood = self.get_tag_loss(encoder_out.encoder_out, masked_tokens, tagging_head_name, tags)
        return log_likelihood

    def decode(self, src_tokens, masked_tokens=None, tagging_head_name=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        tags = self.get_tag_output(encoder_out.encoder_out, masked_tokens, tagging_head_name)
        return tags


@register_model('transformer_pretrain_prediction')
class TransformerPredictionModel(TransformerMultitaskModel):
    """ Bidirectional Language Model for Sentence Prediction """

    def forward(self, src_tokens, features_only=True, classification_head_name=None, **kwargs):
        encoder_out = self.encoder(src_tokens, **kwargs)
        encoder_feature = encoder_out.encoder_out.transpose(0, 1)  # T x B x C -> B x T x C
        cls_out = self.get_cls_output(encoder_feature, classification_head_name=classification_head_name)
        return cls_out, None


class TransformerForConditionalGeneration(TransformerMultitaskModel):
    """ endoder-decoder
    BartForConditionalGeneration https://github.com/huggingface/transformers/blob/155288f04ba9a5d0a0e4d5be4f6d4e808ad8cfff/src/transformers/modeling_bart.py#L940
    PegasusForConditionalGeneration https://github.com/huggingface/transformers/blob/155288f04ba9a5d0a0e4d5be4f6d4e808ad8cfff/src/transformers/modeling_pegasus.py#L24
    """
    pass


@register_model_architecture('transformer_pretrain', 'transformer_pretrain_base')
def transformer_pretrain_base(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    mass_base(args)


@register_model_architecture('transformer_pretrain', 'transformer_pretrain_tiny')
def transformer_tiny(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    mass_tiny(args)


@register_model_architecture('transformer_pretrain_tagging', 'transformer_pretrain_tagging_base')
def transformer_tagging_base(args):
    transformer_pretrain_base(args)


@register_model_architecture('transformer_pretrain_prediction', 'transformer_pretrain_prediction_base')
def transformer_prediction_base(args):
    transformer_pretrain_base(args)


@register_model_architecture('transformer_pretrain_lm', 'transformer_pretrain_lm_base')
def transformer_tagging_base(args):
    transformer_pretrain_base(args)
