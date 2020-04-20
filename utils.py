import numpy as np 
import os, sys
import glob
from nltk.stem.porter import PorterStemmer
from os.path import isfile
import re
from collections import defaultdict
# re.split('\W+'): chia text ra thành các từ(bỏ đi các kí tự đặc biệt, dấu cách chỉ còn lại số và chữ)
# stem: lay tu goc cua cac tu: trainer, driver = train, drive
def gather_data():
    path = './data_set/'
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path)]
    # dirs = glob.glob(path)
    # print(dirs)
    train_path, test_path = dirs[1], dirs[0]
    list_newgroups = [newgroups for newgroups in os.listdir(train_path)]
    list_newgroups.sort()
    return train_path, test_path, list_newgroups
    # print(list_newgroups)

with open('./data_set/stop_word.txt') as f:
    stop_words = f.read().splitlines()
stemmer = PorterStemmer()

def collect_data_from(parent_dir, newgroup_list):
    data = []
    for group_id, newgroup in enumerate(newgroup_list):
        print(str(group_id) + ' group process!')
        label = group_id
        dir_path = parent_dir + '/' + newgroup + '/'
        files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if isfile(dir_path + filename)]
        files.sort() # filename incluse filename and path to filename
        # print(files)
        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                content = ' '.join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + '<fff>' + filename + '<fff>' + content)
    return data

def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. /df)
    def second_element(x):
        return x[1]
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int) # count number of appearance of each word in vacab
    corpus_size = len(lines)
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split())) # word: vacab of text(text la van ban sau xu li cua cac file)
        for word in words:
            doc_count[word] += 1
    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                  for word, document_freq in zip(doc_count.keys(), doc_count.values()) 
                  if document_freq > 10 and not word.isdigit()] 
    words_idfs.sort(key = second_element, reverse = True) # sort theo -idf(mac dinh tu be den lon) nen idf tu lon den be
    print('Vocabulary size: {}'.format(len(words_idfs)))
    with open('./data_set/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))

def get_tf_idf(data_path, save_data_path):
    with open('./data_set/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                     for line in f.read().splitlines()]
        word_ID = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)
    
    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]),
                     int(line.split('<fff>')[1]),
                     line.split('<fff>')[2])
                     for line in f.read().splitlines()]
        data_tf_idf = []
        for document in documents:
            label, doc_id, text = document
            # print(label)
            words = [word for word in text.split() if word in idfs]
            word_set = list(set(words)) # word set cua tung van ban
            max_term_freq = max([words.count(word) for word in word_set])
            words_tfidfs = []
            sum_squares = 0.0

            for word in word_set:
                    term_freq = words.count(word)
                    tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
                    words_tfidfs.append((word_ID[word], tf_idf_value))
                    sum_squares += tf_idf_value ** 2

            words_tfidfs_normalized = [str(index) + ':' + str(tf_idf_value / np.sqrt(sum_squares))
                                      for index, tf_idf_value in words_tfidfs]

            sparse_rep = ' '.join(words_tfidfs_normalized)
            data_tf_idf.append((label, doc_id, sparse_rep))
            print(len(data_tf_idf))
    
    with open(save_data_path, 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for (label, doc_id, sparse_rep) in data_tf_idf]))