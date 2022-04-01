#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model import Model, TOKENIZER_CLASSES
from train import Trainer
from utils.utils_dataloader import load_data, set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    ############# Set Dirs ############
    model_dir = os.path.join(args.save_dir, "models")
    rationale_dir = os.path.join(args.save_dir, "rationale.npz")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    ############# Set Configs ############
    with open(os.path.join(args.configs), 'r') as fp:
        configs = json.load(fp)
    k = args.length
    SEED = int(args.seed)
    set_seed(SEED)


    ############ Load Dataset ############
    print("Loading Dataset...")
    tokenizer_class = TOKENIZER_CLASSES[configs["model_params"]["model_type"]]
    tokenizer = tokenizer_class.from_pretrained(configs["model_params"]["model_type"], do_lower_case=True)
    train_dataloader, valid_dataloader, test_dataloader = load_data(args.data_dir, configs['data_params'], tokenizer, SEED)


    ############# Load Model ############
    print('Loading Model...')
    model_kwargs = configs['model_kwargs']
    model = Model(configs, k, SEED, **model_kwargs).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= LimitedInk Modeling =============')
    print(f"\t ====== Model Setting: length percent={k}, seed={SEED} ====== ")
    print(f"\t ====== Model Setting: model_kwargs={model_kwargs} ====== ")
    print(f"\t ====== Configs Setting: model_kwargs={configs} ====== ")
    print('\t-----------------------------------------------------------------------')


    ############# Training Setting ############
    epochs = configs['train_params']['epochs']
    total_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr = configs['train_params']['lr'], eps = 1e-8)  # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)
    trainer = Trainer(configs, SEED)


    ############# Resume Model ############
    if args.resume:
        print('\t-----------------------------------------------------------------------')
        print(f'\t ======= Resuming model from checkpoint. Evaluating on Testset =======')
        print('\t-----------------------------------------------------------------------')
        assert os.path.isdir(model_dir), 'Error: no checkpoint directory found!'
        model.load_model(model_dir, tokenizer)
        trainer.evaluate(epochs, model, test_dataloader, tokenizer, None, None)
        exit()


    ############# Start Training ############
    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= LimitedInk Model Training =============')
    print('\t-----------------------------------------------------------------------')
    start_time = time.time()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f'\t ========================== Epoch: {epoch+1:02} ==========================')
        trainer.train(epoch, model, train_dataloader, optimizer, scheduler)
        trainer.evaluate(epoch, model, valid_dataloader, tokenizer, model_dir)
    end_time = time.time()
    print("Epoch{}: Model: {} - Total Time: {} sec".format(epoch, model_dir, str(end_time-start_time)))


    ############# Start Evaluation #############
    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= LimitedInk Model Evaluating =============')
    print('\t-----------------------------------------------------------------------')
    model.load_model(model_dir, tokenizer)
    trainer.evaluate(epoch, model, test_dataloader, tokenizer, None, rationale_dir=rationale_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default='../data/movies')
    parser.add_argument('--save_dir', dest='save_dir', default='../checkpoints/movies/bert/token_rationale/length_level_0.5/seed_1234')
    parser.add_argument('--configs', dest='configs', default='../limitedink/model/params/movies_configs_token.json')
    parser.add_argument('--length', type = float, default=0.5, help='The percentage of rationale length.')
    parser.add_argument('--seed', type = str, default=1234, help='seed')
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    args = parser.parse_args()
    main(args)