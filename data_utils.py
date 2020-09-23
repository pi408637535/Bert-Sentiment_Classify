import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
import pickle as pkl
from tqdm import tqdm

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def read_examples(input_file, is_training):
    '''将input_file中的每一行转化为一个python对象，并append到examples列表中

    '''
    examples = []
    if "csv" in input_file:
        df = pd.read_csv(input_file)

        '''
        for index,data in df.iterrows():
            #text_a对应content, text_b对应title
            examples.append(InputExample(guid=data["ID"], text_a=data["txt"], label=data["Label"]))
        '''
        for val in df[['id', 'content', 'title', 'label']].values:
            # text_a对应content, text_b对应title
            examples.append(InputExample(guid=val[0], text_a=val[1], text_b=val[2], label=val[3]))


    elif "txt" in input_file:

        with open(input_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue

                # Todo
                if len(lin.split('\t')) != 2:
                    continue

                content, label = lin.split('\t')
                examples.append(InputExample(guid="random", text_a=content, label=int(label)))

    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_single(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        else:
            tokens_a.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length, split_num, is_training):
    features = []

    for example_index, example in enumerate(examples):

        context_tokens = tokenizer.tokenize(example
                                            .text_a)
        ending_tokens = tokenizer.tokenize(example.text_b)

        skip_len = len(context_tokens) / split_num
        choices_features = []

        # 每次选择一小段文本
        for i in range(split_num):
            context_tokens_choice = context_tokens[
                                    int(i * skip_len): int((i + 1) * skip_len)]  # split_num：将文本分割成split_num段
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + ending_tokens + ["[SEP]"] + context_tokens_choice + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if example_index < 1 and is_training:
                print("*** Example ***")
                print("idx: {}".format(example_index))
                print("guid: {}".format(example.guid))
                print("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                print("input_ids: {}".format(' '.join(map(str, input_ids))))
                print("input_mask: {}".format(' '.join(map(str, input_mask))))
                print("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                print("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label
            )
        )
    return features



def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]



def load_dataset(train_examples, tokenizer, max_seq_length, split_num, batch_size):
    train_features = convert_examples_to_features(
        train_examples, tokenizer, max_seq_length, split_num, True)
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    train_sampler = RandomSampler(train_data)
    data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return data_loader

def build_dataset(config, args):
    if os.path.isfile(config.datasetpkl):
        dataset = pkl.load( open(config.datasetpkl, "rb" ))
        train = dataset['train']
        dev = dataset['dev']
    else:
        train_examples = read_examples(config.train_path, is_training=True)
        train_features = load_dataset(train_examples, config.tokenizer, args.max_seq_length,
                                  args.split_num, args.batch_size)

        dev_examples = read_examples(config.dev_path, is_training=True)
        dev_features = load_dataset(dev_examples, config.tokenizer, args.max_seq_length,
                                      args.split_num, args.batch_size)

        dataset = {}
        dataset['train'] = train_features
        dataset['dev'] = dev_features

        pkl.dump(dataset, open(config.datasetpkl, "wb"))

        train,dev = train_features,dev_features

    return train, dev,  # test





