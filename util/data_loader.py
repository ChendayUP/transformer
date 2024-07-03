import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class DataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = get_tokenizer(tokenize_en)
        self.tokenize_de = get_tokenizer(tokenize_de)
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def yield_tokens(self, data_iter, language):
        for data in data_iter:
            if language == 'de':
                yield self.tokenize_de(data[0])
            else:
                yield self.tokenize_en(data[1])

    def make_dataset(self):
        train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=self.ext)
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source_vocab = build_vocab_from_iterator(self.yield_tokens(train_data, 'de'), min_freq=min_freq, specials=[self.init_token, self.eos_token, '<unk>', '<pad>'])
        self.target_vocab = build_vocab_from_iterator(self.yield_tokens(train_data, 'en'), min_freq=min_freq, specials=[self.init_token, self.eos_token, '<unk>', '<pad>'])
        self.source_vocab.set_default_index(self.source_vocab['<unk>'])
        self.target_vocab.set_default_index(self.target_vocab['<unk>'])

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(torch.tensor([self.source_vocab[self.init_token]] + self.source_vocab(self.tokenize_de(src_sample)) + [self.source_vocab[self.eos_token]], dtype=torch.long))
            tgt_batch.append(torch.tensor([self.target_vocab[self.init_token]] + self.target_vocab(self.tokenize_en(tgt_sample)) + [self.target_vocab[self.eos_token]], dtype=torch.long))
        src_batch = pad_sequence(src_batch, padding_value=self.source_vocab['<pad>'])
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.target_vocab['<pad>'])
        return src_batch, tgt_batch

    def make_iter(self, train, validate, test, batch_size, device):
        print('batch_size', batch_size)
        train_iterator = DataLoader(train, batch_size=batch_size, collate_fn=self.collate_fn)
        valid_iterator = DataLoader(validate, batch_size=batch_size, collate_fn=self.collate_fn)
        test_iterator = DataLoader(test, batch_size=batch_size, collate_fn=self.collate_fn)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator

