from __future__ import  division, print_function

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import io


class BertClassificationModel:
    def __init__(self, n_epochs=4, batch_size=10, lr=2e-5):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.MAX_LEN = 128
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device('cuda:0')
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.lr = lr

    def _preprocess_batch(self, x):
        return [self.__preprocess_sent(sent) for sent in x]

    def __preprocess_sent(self, x):
        sent = "[CLS] " + x + " [SEP]"
        tokenized_sent = self.tokenizer.tokenize(sent)
        ids_sent = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        while len(ids_sent) < self.MAX_LEN:
            ids_sent.append(0)
        return ids_sent

    def __make_attention_masks(self, x):
        attention_masks = []
        for seq in x:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def __fit_net(self, train_dataloader):
        self.model.cuda()

        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=self.lr, warmup=.1)

        train_loss_set = []

        for _ in trange(self.n_epochs, desc="Epoch"):
            self.model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, = batch
                optimizer.zero_grad()

                loss = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
                train_loss_set.append(loss.item())

                loss.backward()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            # add Validation

            self.model.eval()

            #todo - add validation acc

    def fit(self, x, y):
        padded_ids = [self.__preprocess_sent(xi) for xi in x]
        attention_masks = self.__make_attention_masks(padded_ids)

        train_inputs = torch.tensor(padded_ids)
        train_masks = torch.tensor(attention_masks)
        train_labels = torch.tensor(y)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        self.__fit_net(train_dataloader)

    def predict_batch_proba(self, x):
        padded_ids = [self.__preprocess_sent(xi) for xi in x]
        attention_masks = self.__make_attention_masks(padded_ids)

        pred_inputs = torch.tensor(padded_ids)
        pred_masks = torch.tensor(attention_masks)

        data = TensorDataset(pred_inputs, pred_masks)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        self.model.eval()

        res = []

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_ids, b_mask = batch
            with torch.no_grad():
                logits = self.model(b_ids, token_type_ids=None, attention_mask=b_mask)

            logits = logits.detach().cpu().numpy()
            res.append(logits)

        return res

    def predict_one(self, x):
        return self.predict_batch_proba(x)
