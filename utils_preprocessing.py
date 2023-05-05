import re
from stop_words import get_stop_words

# handle stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import numpy as np

# count word occurrence
from collections import Counter
from itertools import chain


# remove stopwords. (SLOW)
def remove_stopwords(text):
    # my own stop words. Not much a difference
    #stop_words = nltk.corpus.stopwords.words("english")

    # stop words used in the paper
    stop_words = get_stop_words('english')

    word_token = nltk.tokenize.word_tokenize(text)
    text = " ".join(w for w in word_token if not w in stop_words)
    return text


# remove punctuations (special characters) and digits
def remove_special(text):
    #word_token = nltk.tokenize.word_tokenize(text)
    #text =  " ".join(w for w in word_token)
    text = re.sub("[^A-Za-z]+", " ", text) # do not use [^A-Za-z]\s

    # remove cancatnated digits
    #remove_digits = str.maketrans('', '', digits)
    #text = text.translate(remove_digits)
    return text
    
# convert dataframe to dictionary
def df_to_dict(words):
    # this funtion builds a dict, by giving each word a unique value
    # words: numpy array
    # output: dictionary 1, key is a unique word, value is an arbitary integer number
    #         dictionary 2, key is the arbitary integer number, value is the unique word

    dict = {}
    dict[''] = 0
    val = 1
    
    dict_rev = {}
    dict_rev[0] = ''
    
    for i in range(words.shape[0]):
        #print('Current row: ', i)
        for j in range(words.shape[1]):
            # skip empty string
            if(words[i,j] == ''):
                continue
            
            if(not words[i,j] in dict.keys()):
                #print(words[i,j])
                dict[words[i,j]] = val
                
                # build the reverse dict
                dict_rev[val] = words[i,j]
                
                # increment val
                val += 1
                #print(words[i,j], val)
    return dict, dict_rev
    
# convert word in form of string to in form of integer
def dict_to_array(x_words, x_words_dict):
    # map matrix of string to matrix of index, by following the dictionary x_words_dict
    x = np.zeros(x_words.shape)
    
    for i in range(x.shape[0]):
        #print('i is: ', i)
        for j in range(x.shape[1]):
            x[i,j] = x_words_dict[x_words[i,j]]
    
    return x.astype(int)

# remove low occurrence in a data frame    
def remove_low_occurrence(dataframe_text, min_occurrence):
    # split words into lists
    v = dataframe_text.str.split().tolist()
    # compute global word frequency
    c = Counter(chain.from_iterable(v))
    # filter, join, and re-assign
    dataframe_text = [' '.join([j for j in i if c[j] >= min_occurrence]) for i in v]
    return dataframe_text


# process wikipedia page
def process_wiki(wiki_file):
    # input is string, meaning the path to wikipedia file
    # return a dictionary, key is ICD-9, value is concatenated descriptions
    
    with open(wiki_file, encoding='UTF-8') as f:
        lines = f.readlines()

    wiki_dic = {}
    descriptions = str()
    # build a dictionary, key (str) is ICD code, value (str) is text
    for line in lines:
        # skip empty lines
        if(len(line) == 0):
            continue

        if('XXXdiseaseXXX' in line): # the start of a code
            # each description may correspond to MULTIPLE ICD codes
            icds = []
            disease_set = []
            # parse the line, and find out the ICD-9 code
            words = line.split()
            for word in words:
                if ('d_' in word):
                    icds.append(word[2:])
                    # break # if only count the first ICD-9 code
                elif(not word == 'XXXdiseaseXXX'): # this word is part/whole of disease
                    disease_set.append(word)
        elif('XXXendXXX' in line): # the end of a code
            # need to store for each code in icd_set
            for code in icds:
                if(code in wiki_dic.keys()):
                    #print(code)
                    wiki_dic[code] += ' ' + descriptions
                else:
                    wiki_dic[code] = descriptions
        
            # clear descriptions
            descriptions = str()
            icds = {}
            disease_set = {}
        else: # the body of code
            descriptions += ' '+ line.strip()
        
    return wiki_dic
    
# calculate code frequency group
def code_group_f(dataframe_icdcode, code_occurrence):
    # analyze the percentage of code with occurence of [1-10, 11-50, 51-100, 101-500, >500]
    # return code in each of the five groups
    
    # Verify label frequency distribution
    total_occurrence = dataframe_icdcode.apply(lambda x: len(x)).sum() 
    
    # frequency_occurrence: [1-10, 11-50, 51-100, 101-500, >500]
    frequency_occurrence = np.array([0,0,0,0,0])
    percentage_occurrence = np.array([0,0,0,0,0])

    # partition code into five groups, based on code occurrence
    code_group = [[],[],[],[],[]]

    for key in code_occurrence.keys():
        if(code_occurrence[key] <= 10): # this code has occurrence no more than 10
            frequency_occurrence[0] += 1
            percentage_occurrence[0] += code_occurrence[key]
            code_group[0].append(key)
        elif(code_occurrence[key] <= 50 ):
            frequency_occurrence[1] += 1
            percentage_occurrence[1] += code_occurrence[key]
            code_group[1].append(key)
        elif(code_occurrence[key] <= 100):
            frequency_occurrence[2] += 1
            percentage_occurrence[2] += code_occurrence[key]
            code_group[2].append(key)
        elif(code_occurrence[key] <= 500):
            frequency_occurrence[3] += 1
            percentage_occurrence[3] += code_occurrence[key]
            code_group[3].append(key)
        else:
            frequency_occurrence[4] += 1
            percentage_occurrence[4] += code_occurrence[key]
            code_group[4].append(key)

    print('For frequency range [1,10], [11,50], [51,100], [101,500], [501,+00]:')
    print('My code has: ', frequency_occurrence.tolist())
    print('The paper has: [80, 73, 25, 82, 84]')

    print('My percentage of code occurrence is: ', np.around(percentage_occurrence/total_occurrence * 100, decimals = 1))
    print('The paper has percentage of code occurrence of: 0.1, 0.6, 0.6, 6.7, 92.0')
    
    return code_group


