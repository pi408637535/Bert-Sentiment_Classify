# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
#from pytorch_pretrained_bert.optimization import BertAdam
from transformers.optimization import AdamW
#from pytorch_pretrained.optimization import BertAdam
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    #LayerNorm,bias是不需要decay的
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "bert" in n], "lr":config.bert_learning_rate, 'weight_decay': 0.01 },
        {'params': [p for n, p in param_optimizer if "bert" not in n], "lr": config.other_learning_rate,'weight_decay': 0.01 },

    ]

    # LayerNorm,bias是不需要decay的
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': config.weight_decay},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate,  warmup=0.05,t_total=len(train_iter) * config.num_epochs)


    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_learning_rate,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False


    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05,
                                                num_training_steps=len(train_iter) * config.num_epochs)  # PyTorch scheduler

    #optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, data in enumerate(train_iter):
            data = tuple(t.to(config.device) for t in data)
            input_ids, input_mask, segment_ids, label_ids = data#
            outputs = model(data)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label_ids)
            loss.backward()

            optimizer.step()
            scheduler.step()


            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = label_ids.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    #test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        #for texts, labels in data_iter:
        for i, data in enumerate(data_iter):
            data = tuple(t.to(config.device) for t in data)
            input_ids, input_mask, segment_ids, label_ids = data  #
            outputs = model(data)
            loss = F.cross_entropy(outputs, label_ids)
            loss_total += loss
            labels = label_ids.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)