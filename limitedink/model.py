#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
from transformers import (BertTokenizer, BertConfig, BertModel, \
                          DistilBertTokenizer, DistilBertConfig, DistilBertModel,  \
                          RobertaTokenizer, RobertaConfig, RobertaModel )

from classifiers.modeling_distilbert import DistilBertForSequenceClassification
from classifiers.modeling_bert import BertForSequenceClassification
from classifiers.modeling_roberta import RobertaForSequenceClassification
from utils.utils_dataloader import set_seed
from modules import Model_Loss, Sparse_Loss, Control_Loss

import warnings
warnings.filterwarnings("ignore")


TOKENIZER_CLASSES = {
    'bert-base-uncased': BertTokenizer,
    'roberta-base': RobertaTokenizer,
    "distilroberta-base": RobertaTokenizer,
    'distilbert-base-uncased' : DistilBertTokenizer
}


MODEL_CLASSES = {
    'bert-base-uncased': (BertConfig, BertForSequenceClassification, BertModel),
    'roberta-base': (RobertaConfig, RobertaForSequenceClassification, RobertaModel),
    "distilroberta-base": (RobertaConfig, RobertaForSequenceClassification, RobertaModel),
    'distilbert-base-uncased' : (DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel)
}


MODEL_LOSSES = {
    "limitedink": Model_Loss,
    "sparse": Sparse_Loss,
    "control": Control_Loss
}


EPS = 1e-16
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, config_params, k, SEED, embedding_dim=768, **model_kwargs):
        super(Model, self).__init__()
        set_seed(SEED)
        self.k = k
        self.tau = config_params['model_params']['tau']
        self.rationale_level = config_params['data_params']['rationale_level']
        self.remove_query_input = config_params['data_params']['remove_query_input']

        config_class, classifier_class, encoder_class = MODEL_CLASSES[config_params['model_params']['model_type']]
        self.identifier = _Identifier(self.k, self.tau, self.rationale_level, embedding_dim, encoder_class, config_params).to(device)
        self.classifier = _Classifier(self.k, config_class, classifier_class, config_params)
        
        loss_class = MODEL_LOSSES[config_params['model_params']['loss_function']]
        self.loss = loss_class(k, num_classes=config_params['model_params']['num_labels'], **model_kwargs)

        # ADD for Sentence
        self.fixed_mask_length = config_params['data_params']['max_query_length'] - 1


    def forward(self, batch):

        if self.rationale_level == "token":
            self.rationale_mask = self.identifier(batch)   # (batch_size, sent_len): (64, 133)
            self.rationale_token_mask = self.rationale_mask
            self.position_mask = 1 - batch[4] if self.remove_query_input else batch[2]

        elif self.rationale_level == "sentence":
            input_mask, p_mask = batch[2], batch[4]
            self.rationale_mask = self.identifier(batch)                                                           # (batch_size, sent_len): (3, 36)
            token_mask_prob = self.rationale_mask.squeeze(1).gather(dim=1, index=p_mask)                                    # token_mask_prob = torch.Size([3, 509])   p_mask
            fixed_mask = torch.ones((token_mask_prob.shape[0], self.fixed_mask_length), dtype=torch.float, device=device)   # shape: torch.Size([3, 3])
            self.rationale_token_mask = torch.cat((fixed_mask, token_mask_prob), dim=-1) *input_mask.float()                 # shape = torch.Size([4, 512])
            self.position_mask = batch[7]

        outputs_sufficiency = self.classifier(self.rationale_token_mask, batch)
        self.outputs_comprehensive = self.classifier((1-self.rationale_token_mask), batch)
        return outputs_sufficiency


    def loss_fn(self, outputs_sufficiency, targets):
        total_loss = self.loss(outputs_sufficiency, self.outputs_comprehensive, targets, self.position_mask, self.rationale_mask)
        return total_loss


    def save_model(self, model_dir, tokenizer):
        identifier_save_dir = os.path.join(model_dir, 'identifier_ckpt_k{}.pt'.format(self.k))
        classifier_save_dir = os.path.join(model_dir, 'classifier_ckpt_k{}.pt'.format(self.k))
        torch.save(self.identifier.state_dict(), identifier_save_dir)
        torch.save(self.classifier.state_dict(), classifier_save_dir)
        tokenizer.save_pretrained(model_dir)


    def load_model(self, model_dir, tokenizer):
        identifier_save_dir = os.path.join(model_dir, 'identifier_ckpt_k{}.pt'.format(self.k))
        classifier_save_dir = os.path.join(model_dir, 'classifier_ckpt_k{}.pt'.format(self.k))
        self.classifier.load_state_dict(torch.load(classifier_save_dir))
        self.identifier.load_state_dict(torch.load(identifier_save_dir))
        tokenizer = tokenizer.from_pretrained(model_dir)
        return tokenizer



class _Identifier(nn.Module):
    def __init__(self, k, tau, rationale_level, embedding_dim, encoder_class, config_params):
        super(_Identifier, self).__init__()
        self.k = k
        self.tau = tau
        self.rationale_level = rationale_level
        self.encoder = encoder_class
        self.remove_query_input = config_params['data_params']['remove_query_input']
        self.dropout = nn.Dropout(p=0.2)

        self._generate_logits = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, 2),
            nn.Softplus())

        self.encoder = encoder_class.from_pretrained(config_params['model_params']['model_type']).to(device)
       

    def _gumbel_sparsity_control(self, input):                                                                                     # input:   torch.Size([bs, max_seq_length, 1])
        """
        Define the Gumble Sampling process to control sparsity.
        """
        input_ = input.unsqueeze(1)                                                                                                # input_:  torch.Size([bs, 1, max_seq_length, 1])
        uniform = torch.rand(size = (input_.size(0), math.ceil(input.size(1) * self.k), input_.size(-2), input_.size(-1))).to(device)    # uniform: torch.Size([bs, rationale_sparity, max_seq_length, 1])
        gumbel = - torch.log(-torch.log(uniform))                                                                                  # same          
        noisy_logits = (gumbel + input_)/self.tau                                                                                  # save 
        samples = nn.Softmax(dim=-2)(noisy_logits)                                                                                 # save 
        _samples = torch.max(samples, dim=1)[0]                                                                                    # _samples: torch.Size([bs, max_seq_length, 1])
        return _samples.view_as(input)


    def forward(self, batch):
        """
        p_mask:  [CLS] and [Paragraph] tokens = 0, [SEP] and [Query] tokens = 1 
        """


        if self.rationale_level == "token":

            inputs = {"input_ids": batch[0], 
                      "labels": batch[1],
                      "attention_mask": batch[2],
                      "p_mask": batch[4]}  

            if self.remove_query_input:
                doc_mask = 1 - inputs['p_mask']
                remove_query_lenth = torch.sum(inputs['attention_mask'][0]) - torch.sum(doc_mask[0]) - 1
                output_embs = self.encoder(inputs['input_ids'], attention_mask = doc_mask)
                sequence_output = output_embs[0] *  doc_mask.unsqueeze(-1).float()                                     # torch.Size([bs, max_seq_length, embedding_dim]) = torch.Size([3, 512, 768])
                sequence_output = self.dropout(sequence_output)
                logits = self._generate_logits(sequence_output.repeat(1,1,2))                                                          # input = torch.Size([bs, max_seq_length, 2 * embedding_dim]);
                                                                                                                                    # output = torch.Size([bs, max_seq_length, 1]);
                batch_max_seq_length = torch.max(torch.sum(inputs["attention_mask"], dim=-1, keepdim=True))                            # truncate logits and then pads (for short inputs)

                truncated_logits = torch.cat((logits[:, 0:1, :], logits[:, (remove_query_lenth+1):batch_max_seq_length, :]), axis=1)

                truncated_outputs = self._gumbel_sparsity_control(truncated_logits)
                outputs = torch.zeros(logits.shape).to(device)

                outputs[:, 0:1, :] = truncated_outputs[:, 0:1, :]
                outputs[:, remove_query_lenth+1:batch_max_seq_length, :] = truncated_outputs[:, 1:, :]
                
                return outputs[:,:,1]

            else:
                output_embs = self.encoder(inputs['input_ids'], attention_mask= inputs['attention_mask'])
                sequence_output = output_embs[0] *  inputs['attention_mask'].unsqueeze(-1).float()                                     # torch.Size([bs, max_seq_length, embedding_dim]) = torch.Size([3, 512, 768])
                sequence_output = self.dropout(sequence_output)

                logits = self._generate_logits(sequence_output.repeat(1,1,2))                                                          # input = torch.Size([bs, max_seq_length, 2 * embedding_dim]);
                                                                                                                                    # output = torch.Size([bs, max_seq_length, 1]);
                batch_max_seq_length = torch.max(torch.sum(inputs["attention_mask"], dim=-1, keepdim=True))                            # truncate logits and then pads (for short inputs)
                truncated_logits = logits[:, :batch_max_seq_length, :]
                truncated_outputs = self._gumbel_sparsity_control(truncated_logits)
                outputs = torch.zeros(logits.shape).to(device)
                outputs[:, :batch_max_seq_length, :] = truncated_outputs                   # torch.Size([bs, max_seq_length, 1]) = torch.Size([3, 512, 1])
                return outputs[:,:,1]




        elif self.rationale_level == "sentence":

            inputs = {"input_ids": batch[0],
                      "labels": batch[1],
                      "attention_mask": batch[2],
                      "sentence_starts": batch[5],
                      "sentence_ends": batch[6],
                      "sentence_mask": batch[7]
                      }

            output_embs = self.encoder(inputs['input_ids'], attention_mask= inputs['attention_mask'])
            sequence_output = output_embs[0] *  inputs['attention_mask'].unsqueeze(-1).float()                                                                 # torch.Size([bs, max_seq_length, embedding_dim]) = torch.Size([3, 512, 768])
            sequence_output = self.dropout(sequence_output)

            sentence_rep_shape = (sequence_output.shape[0], inputs['sentence_starts'].shape[1], sequence_output.shape[-1])                                      # torch.Size([bs, sent_num, embedding_dim]) = torch.Size([3, 512, 768])
            sentence_representations = torch.cat((sequence_output.gather(dim=1, index=inputs['sentence_starts'].unsqueeze(-1).expand(sentence_rep_shape)), \
                                    sequence_output.gather(dim=1, index=inputs['sentence_ends'].unsqueeze(-1).expand(sentence_rep_shape))), dim=-1)             # torch.Size([bs, sent_num, embedding_dim * 2]) = torch.Size([3, 512, 768 * 2])

            logits = self._generate_logits(sentence_representations)                                                                                            # shape = torch.Size([4, 10, 1])
            outputs = self._gumbel_sparsity_control(logits)                                                                                                     # probs: [batch_size, max_num_sentences, z_dim] : # shape = torch.Size([4, 384, 1]) 
            outputs = outputs * inputs['sentence_mask'].unsqueeze(-1).float()
            return outputs[:,:,1]



class _Classifier(nn.Module):
    def __init__(self, k, config_class, classifier_class, config_params):
        super(_Classifier, self).__init__()
        self.k = k
        self.config_params = config_params

        self.config = config_class.from_pretrained(config_params['model_params']['model_type'])
        self.config.num_labels = config_params['model_params']['num_labels']
        self.config.seq_classif_dropout = config_params['model_params']['dropout']
        self.config.max_query_length = config_params['data_params']['max_query_length']
        self.model = classifier_class.from_pretrained(config_params['model_params']['model_type'], config=self.config).to(device)

    def forward(self, rationale_token_mask, batch):
        emd_inputs = {"input_ids": batch[0],
                      "labels": batch[1],
                      "attention_mask": rationale_token_mask * batch[2]}
        outputs = self.model(**emd_inputs)
        return outputs




