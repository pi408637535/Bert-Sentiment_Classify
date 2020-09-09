#!/bin/bash
source activate py35
nohup                   \
python ../train.py     \
--model bert  \
>> bert_base.log  2>&1 &
