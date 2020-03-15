from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text, label=None):
        """Constructs a InputExample.

        Args:
            id: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.id = id
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, true_tokens_ids, input_tokens_ids, input_mask, segment_ids, masked_lm_labels):
        self.true_tokens_ids = true_tokens_ids
        self.input_tokens_ids = input_tokens_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels


class FinetuneBert:
    def __init__(self, ds_path, model_name, base_model="bert-base-uncased",
                 do_lower_case=True, num_epochs=4):
        self.path = ds_path
        self.num_epochs = num_epochs
        self.save_path = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), str(model_name))
        self.model_name = model_name
        self.batch_size = 32
        self.max_seq_len = 64
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20
        self.max_token = 30000

        bert_model = base_model

        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
        self.model = BertForMaskedLM.from_pretrained(bert_model, cache_dir=cache_dir)
        self.model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(5, 768)
        self.model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    def __save_model(self, model_name):
        save_model_path = os.path.join(self.save_path, str(model_name))
        torch.save(self.model, save_model_path)


    def __get_train_examples_tsv(self, data_dir):
        """See base class."""
        return self.__create_examples(
            FinetuneBert.__read_tsv(data_dir), "train")

    @staticmethod
    def __read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def __create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        print(len(lines))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            id = "%s-%s" % (set_type, i)
            text = line[0]
            label = int(line[-1])
            examples.append(
                InputExample(id=id, text=text, label=label))
        return examples

    # todo - split to subfunc
    def __convert_examples_to_features(self, train_examples, label_list):

        features = []

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        for (ex_index, example) in enumerate(train_examples):
            tokens_str = self.tokenizer.tokenize(example.text)
            segment_id = label_map[example.label]
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_str) > self.max_seq_len - 2:
                tokens_str = tokens_str[0:(self.max_seq_len - 2)]

            cnd_indices = [i for i in range(1, len(tokens_str)+1)]
            tokens_str = ['[CLS]'] + tokens_str + ['[SEP]']
            segment_ids = [segment_id] * len(tokens_str)
            masked_labels = [-1] * self.max_seq_len

            num_to_predict = min(self.max_predictions_per_seq,
                                 max(1, int(round(len(tokens_str) * self.masked_lm_prob))))

            ids_to_mask = np.random.choice(cnd_indices, num_to_predict, replace=False)

            input_tokens = tokens_str[:]

            for index in ids_to_mask:
                masked_token = None
                # 80% of the time, replace with [MASK]
                if np.random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if np.random.random() < 0.5:
                        masked_token = tokens_str[index]
                    # 10% of the time, replace with random word
                    else:
                        # todo - sample from whole dictionary
                        masked_token = tokens_str[cnd_indices[np.random.randint(0, len(cnd_indices) - 1)]]

                masked_labels[index] = self.tokenizer.convert_tokens_to_ids([tokens_str[index]])[0]
                input_tokens[index] = masked_token

            true_tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens_str)
            input_tokens_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            input_mask = [1] * len(input_tokens_ids)

            while len(input_tokens_ids) < self.max_seq_len:
                true_tokens_ids.append(0)
                input_tokens_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            features.append(
                InputFeatures(true_tokens_ids=true_tokens_ids,
                              input_tokens_ids=input_tokens_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              masked_lm_labels=masked_labels))
        return features

    def __model_train(self, train_dataloader, save_every_epoch):
        self.model.cuda()
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1)

        self.model.train()

        for epoch in trange(int(self.num_epochs), desc="Epoch"):
            avg_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                _, input_ids, input_mask, segment_ids, masked_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, masked_ids)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                self.model.zero_grad()
                if (step + 1) % 50 == 0:
                    print("avg_loss: {}".format(avg_loss / 50))
                    avg_loss = 0
            if save_every_epoch:
                save_model_name = self.model_name + "_epoch_" + str(epoch + 1)
                self.__save_model(save_model_name)
            else:
                if (epoch + 1) % 10 == 0:
                    save_model_name = self.model_name + "_epoch_" + str(epoch + 1)
                    self.__save_model(save_model_name)

    def finetune(self, save_every_epoch=False):
        train_examples = self.__get_train_examples_tsv(self.path)

        label_list = [0, 1]

        train_features = self.__convert_examples_to_features(
            train_examples, label_list)


        all_true_tokens_ids = torch.tensor([f.true_tokens_ids for f in train_features], dtype=torch.long)
        all_input_tokens_ids = torch.tensor([f.input_tokens_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_true_tokens_ids, all_input_tokens_ids, all_input_mask,
                                   all_segment_ids, all_masked_lm_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        self.__model_train(train_dataloader, save_every_epoch)
        self.__save_model(self.num_epochs+1)

