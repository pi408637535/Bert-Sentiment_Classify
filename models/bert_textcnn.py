# coding: UTF-8
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = "data/" + dataset + '/train.txt'
        self.dev_path = "data/" + dataset + '/dev.txt'   # 训练集
        #self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        #self.test_path = dataset + '/data/test.txt'                                  # 测试集
        #self.class_list = [x.strip() for x in open( dataset + '/data/class.txt').readlines()]                                # 类别名单

        self.class_list = [0,1,2]

        self.save_path = "data/" + dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 1                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_learning_rate = 5e-5
        self.other_learning_rate = 1e-3

        self.bert_path = './bert_pretrain'
        self.datasetpkl = "pkl/data.pkl"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path + "/bert-base-chinese-vocab.txt")
        self.lstm_hidden = 512
        self.bert_hidden = 768
        self.weight_decay = 0


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        ngram = [1, 2, 3]
        in_channels = 1
        out_channels = 100
        class_num = config.num_classes
        self.filters = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=(ele, config.bert_hidden)) for ele in ngram])
        self.fc = nn.Linear(out_channels * len(ngram), class_num)

    @staticmethod
    def conv_filter(conv, text):
        # text: batch,in_channel,h,w: b,in,seq,dim
        data = conv(text)  # data: batch,out_channel,H,1
        data = t.squeeze(data, dim=-1)  # data: batch,out_channel,H
        data = F.max_pool1d(data, kernel_size=(data.shape[-1]))  # data: batch, channel, 1\
        data = t.squeeze(data, dim=-1)
        return data

    def forward(self, x):
        input_ids = x[0]  # batch,split,seq
        input_mask = x[1]
        segment_ids = x[2]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_input_mask = input_mask.view(-1, input_ids.size(-1))
        flat_segment_ids = segment_ids.view(-1, input_ids.size(-1))

        # _, pooled = self.bert(flat_input_ids, attention_mask=flat_input_mask, token_type_ids= flat_segment_ids,output_hidden_states=False)
        _, pooled = self.bert(input_ids=flat_input_ids, token_type_ids=flat_segment_ids,
                              attention_mask=flat_input_mask)

        batch, split_num = input_ids.shape[0], input_ids.shape[1]

        output = pooled.reshape(batch, split_num, -1)

        text = t.unsqueeze(output, dim=1)
        data = [self.conv_filter(cnn_filter, text) for cnn_filter in self.filters]
        data = t.cat(data, dim=1)
        return self.fc(data)


