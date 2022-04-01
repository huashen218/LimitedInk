import os
import json
import torch
import pickle
import bisect
import random
from tqdm import tqdm
import numpy as np
from itertools import chain
from copy import deepcopy as copy
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils.utils_dataset import *
from utils.utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_dir, configs, tokenizer, SEED):
    """
    We'll take training samples in random order.
    For validation/test samples the order doesn't matter, so we'll just read them sequentially.
    """
    set_seed(SEED)

    # rationale_level = configs['rationale_level'] if 'rationale_level' in configs.keys() else "token"
    rationale_level = configs['rationale_level']
    print(f" === Rationale Level: {rationale_level} === ")

    train_dataset = load_split_data(data_dir, configs, split="train", seed=SEED, tokenizer=tokenizer, partial_train=configs['partial_train'], rationale_level=rationale_level)
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = configs['batch_size'])
    print(f"Loading Training Set -- Sample Size = {len(train_dataset)}")

    val_dataset = load_split_data(data_dir, configs, split="val", seed=SEED, tokenizer=tokenizer, partial_train=1.0, rationale_level=rationale_level)
    val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = configs['batch_size'])
    print(f"Loading Validation Set -- Sample Size = {len(val_dataset)}")

    test_dataset = load_split_data(data_dir, configs, split="test", seed=SEED, tokenizer=tokenizer, partial_train=1.0, rationale_level=rationale_level)
    test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = configs['batch_size'])
    print(f"Loading Test Set -- Sample Size = {len(test_dataset)}")

    return train_dataloader, val_dataloader, test_dataloader




def feature_to_tensor(features, rationale_level="token"):
    
    if rationale_level == "token":
        """
        input_ids:        input index;
        class_labels:     prediction label;
        input_mask:       input token = 1, padding = 0; 
        evidence_masks:   human annotated token = 1, others = 0;
        p_mask:           [CLS] and Paragraph tokens = 0, [SEP] and Query tokens = 1 
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_class_labels = torch.tensor([f.class_label for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_evidence_masks = torch.tensor([f.evidence_mask for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.long)      

        tensorized_dataset = TensorDataset(all_input_ids,
                                           all_class_labels, 
                                           all_input_mask,
                                           all_evidence_masks, 
                                           all_p_mask)

    elif rationale_level == "sentence":
        """
        input_ids:        input index;
        class_labels:     prediction label;
        input_mask:       input token = 1, padding = 0;
        p_mask:           p_mask only contains sentence membership of the paragrph and is of length 512 - max_quey_len
        sentence_starts:  index of sentence start;
        sentence_ends:    index of sentence end;
        sentence_mask:    input sentence = 1, others = 0; sentence_mask doesn't contain query with (max_query_length)
        evidence_masks:   human annotated sentence = 1, others = 0
        """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_class_labels = torch.tensor([f.class_label for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.long)
        all_sentence_starts = torch.tensor([f.sentence_starts for f in features], dtype=torch.long)
        all_sentence_ends = torch.tensor([f.sentence_ends for f in features], dtype=torch.long)
        all_sentence_mask = torch.tensor([f.sentence_mask for f in features], dtype=torch.long)
        all_evidence_masks = torch.tensor([f.evidence_mask for f in features], dtype=torch.long)

        tensorized_dataset = TensorDataset(all_input_ids,
                                           all_class_labels, 
                                           all_input_mask, 
                                           all_evidence_masks,
                                           all_p_mask, 
                                           all_sentence_starts, 
                                           all_sentence_ends, 
                                           all_sentence_mask
                                           )

    return tensorized_dataset



def load_split_data(data_dir, configs, split, seed, tokenizer, partial_train=1.0, rationale_level="token", save_human_annotation=True):
    set_seed(seed)

    cached_features_root = os.path.join(data_dir, rationale_level, split)
    if not os.path.exists(cached_features_root):
        os.makedirs(cached_features_root)

    cached_features_file = os.path.join(cached_features_root, configs['cached_features_file'])    

    if os.path.exists(cached_features_file) and not configs['overwrite_cache']:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        tensorized_dataset = feature_to_tensor(features, rationale_level=rationale_level)
        return tensorized_dataset

    else:

        """Step1: Read Annotations"""
        dataset = annotations_from_jsonl(os.path.join(data_dir, split + ".jsonl"))  # get a list of Annotation
        docids = set(e.docid for e in chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(dataset))))) # length=2915
        
        if "movies" in configs["task"]:
            docids.add("posR_161.txt")

        if "boolq" in configs["task"] or "evidence_inference" in configs["task"]:
            docids  = set([ex.docids[0] for ex in dataset])

        if configs["task"] in ["beer", "imdb", "twitter", "sst"]: # no human annotations
            docids= set([ex.annotation_id for ex in dataset])


        """Step2: Read Full Tokenized Texts"""
        documents = load_documents(data_dir, docids)  # get a list of tokenized docs


        """Step3: Generate Token-wise Examples"""
        examples = read_examples(configs, data_dir, dataset, documents, split)  # get a list of "utils.utils_dataset.Example object"
        examples = examples[:int(partial_train * len(examples))]


        """Step4: Convert Examples to Features"""

        if rationale_level == "sentence":
            print("==> Sentence-Level Rationale Features <==")
            features = convert_examples_to_sentence_features(configs, 
                                                    examples=examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=configs['max_seq_length'],
                                                    max_query_length=configs['max_query_length'])

        elif rationale_level == "token":
            print("==> Token-Level Rationale Features <==")
            features = convert_examples_to_features(configs, 
                                                    examples=examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=configs['max_seq_length'],
                                                    max_query_length=configs['max_query_length'])
            if save_human_annotation:
                human_annotations = {
                    "tokens": np.array([f.tokens for f in features]),
                    "evidence_masks": np.array([f.evidence_mask for f in features]),
                    "class_labels": np.array([f.class_label for f in features]),
                    "input_masks": np.array([f.input_mask for f in features]),
                    "p_masks": np.array([f.p_mask for f in features]),
                    }
                with open(os.path.join(data_dir, 'human_annotations.pickle'), 'wb') as handle:
                    pickle.dump(human_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        torch.save(features, cached_features_file)
        tensorized_dataset = feature_to_tensor(features, rationale_level=rationale_level)

    return tensorized_dataset
