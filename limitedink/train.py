#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils_dataloader import set_seed
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_valid_f1 = 0
log_train_step, log_val_step = 500, 200

class Trainer(object):

    def __init__(self, configs, SEED):
        super(Trainer, self).__init__()
        self.configs = configs
        set_seed(SEED)

    def train(self, epoch, model, dataloader, optimizer, scheduler):
        predictions_labels = []
        true_labels = []
        total_loss = 0
        total_acc = 0
        model.train()

        # for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training..."):
        for batch_idx, batch in enumerate(dataloader):
            true_labels += batch[1].numpy().flatten().tolist()
            batch = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(batch)
            loss = model.loss_fn(outputs, batch[1])
            _, logits = outputs[:2]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits = logits.detach().cpu().numpy()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
            total_loss += loss.item()
            acc = accuracy_score(true_labels, predictions_labels)
            total_acc += acc.item()

            if batch_idx % log_train_step == 0:
                print("\t \tbatch_idx: {} - train_loss: {} - train_acc: {} ".format((batch_idx+1), total_loss / (batch_idx+1), total_acc / (batch_idx+1)))

        train_loss, train_acc = total_loss / len(dataloader), total_acc / len(dataloader)
    
        print('\t-----------------------------------------------------------------------')
        print(f'\t| Epoch: {epoch} | Train Loss: {train_loss:.3f} ')
        print('\t-----------------------------------------------------------------------')

        evaluation_report = classification_report(true_labels, predictions_labels, labels=list(self.configs['data_params']['labels_ids'].values()), target_names=list(self.configs['data_params']['labels_ids'].keys()))
        print(evaluation_report)
        return



    def evaluate(self, epoch, model, dataloader, tokenizer, model_dir, rationale_dir=None):
        global best_valid_acc, best_valid_f1
        best_valid_acc = 0 if epoch == 0 else best_valid_acc
        best_valid_f1 = 0 if epoch == 0 else best_valid_f1

        predictions_labels = []
        true_labels = []
        dataset_attributions = []
        total_loss = 0
        total_acc = 0
        model.eval()
        with torch.backends.cudnn.flags(enabled=False): 
            # for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating..."):
            for batch_idx, batch in enumerate(dataloader):
                true_labels += batch[1].numpy().flatten().tolist()
                batch = tuple(t.to(device) for t in batch)
                outputs = model(batch)
                loss = model.loss_fn(outputs, batch[1])
                _, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                predictions_labels += logits.argmax(axis=-1).flatten().tolist()
                total_loss += loss.item()
                acc = accuracy_score(true_labels, predictions_labels)
                total_acc += acc.item()

                if rationale_dir is not None:
                    # attributions = model.identifier(batch)
                    attributions = model.rationale_token_mask
                    dataset_attributions.append(attributions.cpu().detach().numpy())

                if batch_idx % log_val_step == 0:
                    print("\t \tbatch_idx: {} - val_loss: {} - val_acc: {} ".format((batch_idx+1), total_loss / (batch_idx+1), total_acc / (batch_idx+1)))

        if rationale_dir is not None:
            np.savez(os.path.join(rationale_dir), data=np.array(dataset_attributions), pred=predictions_labels, true=true_labels)

        valid_loss, valid_acc = total_loss / len(dataloader), total_acc / len(dataloader)
        
        print('\t-----------------------------------------------------------------------')
        print(f'\t| Epoch: {epoch} | Validation Loss: {valid_loss:.3f} ')
        print('\t-----------------------------------------------------------------------')


        evaluation_report_dict = classification_report(true_labels, predictions_labels, labels=list(self.configs['data_params']['labels_ids'].values()), target_names=list(self.configs['data_params']['labels_ids'].keys()), output_dict=True)
        valid_f1 = evaluation_report_dict['weighted avg']['f1-score']

        evaluation_report = classification_report(true_labels, predictions_labels, labels=list(self.configs['data_params']['labels_ids'].values()), target_names=list(self.configs['data_params']['labels_ids'].keys()))
        print(evaluation_report)

        if model_dir is not None:
            if valid_f1 > best_valid_f1:
                model.save_model(model_dir, tokenizer)
                best_valid_f1 = valid_f1

        return



