from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class BertAug:
    def __init__(self, model_name, do_lower_case=True, base_model="bert-base-uncased", use_untuned=False, use_stop=False):
        self.model_name = model_name
        bert_model = base_model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        if use_untuned:
            cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
            self.model = BertForMaskedLM.from_pretrained(bert_model, cache_dir=cache_dir)
            self.model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(5, 768)
            self.model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        else:
            weights_path = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), model_name)
            self.model = torch.load(weights_path)
        self.model.cuda()

        self.MAX_LEN = 10
        self.__segment_proc_flag = True
        if use_stop:
            self.__stop_words = set(stopwords.words('english'))
        else:
            self.__stop_words = []


    def aug_batch(self, sents):
        pass

    @staticmethod
    def __rev_wordpiece(token_str):
        if len(token_str) > 1:
            for i in range(len(token_str) - 1, 0, -1):
                if token_str[i] == '[PAD]':
                    token_str.remove(token_str[i])
                elif len(token_str[i]) > 1 and token_str[i][0] == '#' and token_str[i][1] == '#':
                    token_str[i - 1] += token_str[i][2:]
                    token_str.remove(token_str[i])
        return " ".join(token_str[1:-1])

    def __auto_create_mask_poses(self, tokens, nrandom):
        clear_numbers = []  # numbers of not stop tokens(words)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok not in self.__stop_words and len(tok) > 3 and not tok.startswith('##') \
                    and not tok.startswith('['):
                if i < len(tokens) - 1 and tokens[i + 1].startswith('##'):
                    pass
                else:
                    clear_numbers.append(i)
            i += 1
        nrandom = min(nrandom, len(clear_numbers) / 2)
        mask_poses = np.random.choice(clear_numbers, int(nrandom), replace=False)
        return mask_poses

    def __mask_tokens(self, mask, tokens, nrandom):
        if mask is None:
            mask_poses = self.__auto_create_mask_poses(tokens, nrandom)
            for i in mask_poses:
                tokens[i] = '[MASK]'
        else:
            mask_poses = []
            tokens_i = 0
            mask_i = 0
            while tokens_i < len(tokens) and mask_i < len(mask):
                if tokens[tokens_i].startswith('##'):
                    if mask[mask_i - 1] == 1:
                        tokens[tokens_i] = '[MASK]'
                        mask_poses.append(tokens_i)
                    tokens_i += 1
                    continue
                if mask[mask_i] == 1:
                    if not self.__segment_proc_flag and tokens_i < len(tokens) - 1 \
                            and tokens[tokens_i + 1].startswith('##'):
                        mask[mask_i] = 0
                    else:
                        tokens[tokens_i] = '[MASK]'
                        mask_poses.append(tokens_i)
                tokens_i += 1
                mask_i += 1
        return mask_poses

    def __proc_sent(self, sent, mask, sent_label, nrandom):
        # tokens(with mask); attention; label_seq
        tokens = self.tokenizer.tokenize(sent)
        tokens_pure = ['[CLS]'] + tokens[:] + ['[SEP]']

        masked_poses = self.__mask_tokens(mask, tokens, nrandom)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        attention_seq = [1] * len(tokens)

        input_token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        true_tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens_pure)

        if sent_label:
            label = 1
        else:
            label = 0
        label_seq = [label] * len(input_token_ids)

        while len(input_token_ids) < self.MAX_LEN:
            input_token_ids.append(0)
            attention_seq.append(0)
            label_seq.append(0)

        return true_tokens_ids, input_token_ids, attention_seq, label_seq, masked_poses

    def __bert_eval(self, input_ids, attention_seq, label_seq):
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_tensor = torch.tensor([attention_seq], dtype=torch.long)
        label_tensor = torch.tensor([label_seq], dtype=torch.long)

        self.model.eval()
        input_ids_tensor = input_ids_tensor.cuda()
        attention_tensor = attention_tensor.cuda()
        label_tensor = label_tensor.cuda()

        predictions = self.model(input_ids_tensor, label_tensor, attention_tensor)
        predictions = predictions.detach().cpu()
        return predictions

    @staticmethod
    def __softmax(p):
        tmp = np.exp(p)
        return tmp/np.sum(tmp)

    @staticmethod
    def __temp_softmax(x, temp):
        p = BertAug.__softmax(x)
        arg = np.log(p)/temp
        return BertAug.__softmax(arg)

    def aug_sent(self, sent, label, mask=None, n_random=2, n_samples=4, temp=0.5, n_rounds=4):
        res_sent = {sent}
        max_round_iter = n_samples * 2
        for r in range(n_rounds):
            iter = 0
            gen_sent = set()
            true_tokens_ids, input_tokens_ids, attention_seq, label_seq, masked_ids = self.__proc_sent(sent, mask, label, n_random)
            res_tensor = self.__bert_eval(input_tokens_ids, attention_seq, label_seq)
            while len(gen_sent) < n_samples and iter < max_round_iter:
                new_sent_tokens_ids = true_tokens_ids[:]
                for idx in masked_ids:
                    cnd_val = res_tensor[0][idx+1]
                    cnd_val = cnd_val.cpu().numpy()
                    cnd_val_softed = BertAug.__temp_softmax(cnd_val, temp)
                    cnd = np.random.choice(len(cnd_val_softed), 1, p=cnd_val_softed)
                    new_sent_tokens_ids[idx+1] = int(cnd)
                new_sent_tokens = self.tokenizer.convert_ids_to_tokens(new_sent_tokens_ids)
                new_sent = self.__rev_wordpiece(new_sent_tokens)
                if new_sent not in res_sent:
                    gen_sent.add(new_sent)
                iter += 1
            res_sent += gen_sent
        return list(res_sent)
