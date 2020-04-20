import numpy as np 
import os, sys
import glob
from nltk.stem.porter import PorterStemmer
from utils import *

train_path, test_path, list_newgroups =  gather_data()
print(list_newgroups)

train_data = collect_data_from(parent_dir = train_path,
							   newgroup_list = list_newgroups)

test_data = collect_data_from(parent_dir = test_path,
							   newgroup_list = list_newgroups)

full_data = train_data+ test_data

with open('./data_set/20news_train_processed.txt', 'w') as f:
	f.write('\n'.join(train_data))

with open('./data_set/20news_test_processed.txt', 'w') as f:
	f.write('\n'.join(test_data))

with open('./data_set/20news_full_processed.txt', 'w') as f:
	f.write('\n'.join(full_data))

generate_vocabulary('./data_set/20news_full_processed.txt')

get_tf_idf('./data_set/20news_train_processed.txt', './data_set/train_tf_idf.txt')
get_tf_idf('./data_set/20news_test_processed.txt', './data_set/test_tf_idf.txt')
