import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def subtask2(x_val, wiki_binary, total_vocabulary, embedding_dim, ksi_fc0_obj, ksi_fc1_obj, ksi_fc2_obj,device ):
    '''
    this function builds similarity vector o2. This function must be used inside class RNN.forward()
    INPUT:
    x_val is unique vocabulary value, shape = (batch_size, x_words.shape[1])

    ksi_fc0_obj = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
    ksi_fc1_obj = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
    ksi_fc2_obj = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)
    
    OUTPUT:
    o2: similarity score all batch_size, shape = (batch_size, 344)
    '''
    
    batch_size = x_val.shape[0]
    
    o2 = torch.Tensor().to(device)
    
    for i in range(batch_size):
        # step 1: expand x to include additional words in wiki_binary

        xi = torch.zeros(total_vocabulary) # shape = (1, vocabulary)

        xi[:x_val.shape[1]] = (x_val[i,:x_val.shape[1]] >0).to(torch.int8)
        
        
        # compare with each wiki_binary
        #for j in range(wiki_binary.shape[0]):
        
        # zi shape (344, total_vocabulary) = (total_vocabulary) * (344, total_vocabulary)
        # need to change zi from torch.float64 to torch.float32
        zi = (xi * wiki_binary).float().to(device) 
        #print('wiki_binary: ', wiki_binary)
        #print('zi shape: ', zi.shape)
        #print('zi: ', zi) 
        # embedding
        
        ei = ksi_fc0_obj(zi) # ei shape = (344, embedding_dim)
            
        # attention weights
        alpha_i = ksi_fc1_obj(ei) # alpha_i shape = (344, embedding_dim)  
        v_i = alpha_i * ei # v_i shape = (344, embedding_dim) * (344, embedding_dim)
        
        # similarity score
        s_i = ksi_fc2_obj(v_i) # s_i shape (344,1) =  (344,embedding ) @ (embedding_dim, 1)
        
        # tranpose to have shape of (1, 344)
        s_i = torch.transpose(s_i, 0, 1)
        #print('s_i shape: ', s_i.shape)
    
        # concatenate s_i for all row in batch_size
        o2 = torch.cat((o2, s_i.detach()), dim = 0)  # check dim
    
    return o2
    
def subtask2_attention(x_val, wiki_binary, total_vocabulary, embedding_dim, ksi_fc0_obj, ksi_fc1_obj, ksi_fc2_obj,device ):
    '''
    this function is for ablation study 
    INPUT:
    x_val is unique vocabulary value, shape = (batch_size, x_words.shape[1])

    ksi_fc0_obj = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
    ksi_fc1_obj = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
    ksi_fc2_obj = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)
    
    OUTPUT:
    o2: similarity score all batch_size, shape = (batch_size, 344)
    '''
    
    batch_size = x_val.shape[0]
    
    o2 = torch.Tensor().to(device)
    
    for i in range(batch_size):
        # step 1: expand x
        xi = torch.zeros(total_vocabulary) # shape = (1, vocabulary)
        xi[:x_val.shape[1]] = (x_val[i,:x_val.shape[1]] >0).to(torch.int8)
        
        
        # compare with each wiki_binary
        
        # zi shape (344, total_vocabulary) = (total_vocabulary) * (344, total_vocabulary)
        # need to change zi from torch.float64 to torch.float32
        zi = (xi * wiki_binary).float().to(device) 

        # embedding
        ei = ksi_fc0_obj(zi) # ei shape = (344, embedding_dim)
            
        # attention weights
        #alpha_i = ksi_fc1_obj(ei) # alpha_i shape = (344, embedding_dim)  
        
        # in KSI-attention, v_i is NOT implemented (see Paper Table 7)
        #v_i = alpha_i * ei # v_i shape = (344, embedding_dim) * (344, embedding_dim)
        
        # similarity score
        # in KSI-attention, s_i is calculated based on ei, not v_i
        #s_i = ksi_fc2_obj(v_i) # s_i shape (344,1) =  (344,embedding ) @ (embedding_dim, 1)
        s_i = ksi_fc2_obj(ei) # s_i shape (344,1) =  (344,embedding ) @ (embedding_dim, 1)
        
        # tranpose to have shape of (1, 344)
        s_i = torch.transpose(s_i, 0, 1)

        # concatenate s_i for all row in batch_size
        o2 = torch.cat((o2, s_i.detach()), dim = 0)  # check dim
    
    return o2

def subtask2_intersection(x_val, wiki_binary, total_vocabulary, embedding_dim, ksi_fc0_obj, ksi_fc1_obj, ksi_fc2_obj,device ):
    '''
    this function builds similarity vector o2. This function must be used inside class RNN.forward()
    INPUT:
    x_val is unique vocabulary value, shape = (batch_size, x_words.shape[1])

    ksi_fc0_obj = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
    ksi_fc1_obj = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
    ksi_fc2_obj = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)
    
    OUTPUT:
    o2: similarity score all batch_size, shape = (batch_size, 344)
    '''
    
    batch_size = x_val.shape[0]
    
    o2 = torch.Tensor().to(device)
    
    for i in range(batch_size):
        # step 1: expand x
        xi = torch.zeros(total_vocabulary) # shape = (1, vocabulary)
        xi[:x_val.shape[1]] = (x_val[i,:x_val.shape[1]] >0).to(torch.int8)
        
        # zi shape (344, total_vocabulary) = (total_vocabulary) * (344, total_vocabulary)
        # zi = (xi * wiki_binary).float().to(device) 
        wiki_binary = wiki_binary.float().to(device)
        eq = ksi_fc0_obj(wiki_binary).float().to(device)
        xi = xi.to(device)
        exi = ksi_fc0_obj(xi).float().to(device)


        # embedding
        #ei = ksi_fc0_obj(zi) # ei shape = (344, embedding_dim)
        ei = eq * exi
            
        # attention weights
        alpha_i = ksi_fc1_obj(ei) # alpha_i shape = (344, embedding_dim)  
        v_i = alpha_i * ei # v_i shape = (344, embedding_dim) * (344, embedding_dim)
        
        # similarity score
        s_i = ksi_fc2_obj(v_i) # s_i shape (344,1) =  (344,embedding ) @ (embedding_dim, 1)
        
        # tranpose to have shape of (1, 344)
        s_i = torch.transpose(s_i, 0, 1)
   
        # concatenate s_i for all row in batch_size
        o2 = torch.cat((o2, s_i.detach()), dim = 0)  # check dim
    
    return o2

# this KSI function replace loop with matrix operations, but required more GPU mem
def subtask2_large_gpu_mem(x_val, wiki_binary, total_vocabulary, embedding_dim, ksi_fc0_obj, ksi_fc1_obj, ksi_fc2_obj,device ):
    
    '''
    this function builds similarity vector o2. This function must be used inside class RNN.forward()
    INPUT:
    x_val is unique vocabulary value, shape = (batch_size, x_words.shape[1])
    wiki_binary is torch.tensor()

    ksi_fc0_obj = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
    ksi_fc1_obj = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
    ksi_fc2_obj = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)
    
    OUTPUT:
    o2: similarity score all batch_size, shape = (batch_size, 344)
    
    '''
    batch_size = x_val.shape[0]
    
    num_code = wiki_binary.shape[0]

    
    xi = torch.zeros(batch_size, total_vocabulary) # shape = (batch_size, vocabulary)
    xi[:,:x_val.shape[1]] = (x_val[:,:x_val.shape[1]] >0).to(torch.int8)
    
    # expand xi from shape (batch_size, total_vocabulary) to (batch_size, 344, total_vocabulary)
    xi = xi.unsqueeze(dim = 1).repeat(1, num_code, 1)
    #print('xi shape:', xi.shape)
        
    # zi shape (batch_size, 344, total_vocabulary) = (batch_size, 344, total_vocabulary) * (344, total_vocabulary)
    # need to change zi from torch.float64 to torch.float32
    zi = torch.mul(xi, wiki_binary).float().to(device) 

    # embedding
    ei = ksi_fc0_obj(zi) # ei shape = (batch_size, 344, embedding_dim)
            
    # attention weights
    alpha_i = nn.Sigmoid(ksi_fc1_obj(ei)) # alpha_i shape = (batch_size, 344, embedding_dim)  
    v_i = alpha_i * ei # v_i shape = (batch_size, 344, embedding_dim) * (batch_size,344, embedding_dim)
        
    # similarity score
    s_i = ksi_fc2_obj(v_i) # s_i shape (batch_size, 344,1) = batch_size, (344,embedding ) @ (embedding_dim, 1)
        
    # tranpose to have shape of (batch_size, 344)
    o2 = torch.squeeze(s_i, dim = 2)
    #print('o2 shape: ', o2.shape)

    
    return o2

# this function calculates the weight of each word in clinical note, i.e., lambda_i
def ksi_weight(target_ICD_code, all_vocab_size, ksi_fc0, ksi_fc1, ksi_fc2, wiki_binary, x_words_vocab_rev, y_labels):
    '''
    Purpose: to calculate ksi_weight, i.e., lambda, for a target ICD-9 code
    Input: target_ICD_code: string, such as '250' for diabetes
           all_vocab_size: vocabulary size including clinical notes, and wikipedia
           ksi_fc0: ksi_fc0_obj weights, ei = ksi_fc0_obj(zi)
           ksi_fc1: ksi_fc1_obj weights, alpha_i = ksi_fc1_obj(ei)
           ksi_fc2: ksi_fc2_obj weigths, s_i = ksi_fc2_obj(v_i)
           wiki_binary: ndarry with shape of (344, all_vocab_size)
    Output:
           lambda_i: a list of pair (word, lambda_i), sorted by the descending order in terms of lambda_i
    '''
    # voT is essentially ksi_fc2, shape of (1, embedding_dim = 100)
    

    # assume in a given clinical note N, all words in the vocabulary exists
    xi = torch.ones(all_vocab_size)

    # zi shape = (344, 43325)
    zi = xi * wiki_binary

    # calculate ei, shape of (344, embedding_dim)
    ei = zi.float() @ torch.transpose(ksi_fc0, 0,1)

    # calculate alpha_i, shape of (344, embedding_dim=100)
    alpha_i = ei @ torch.transpose(ksi_fc1, 0, 1)

    # find out the row index corresponding to ICD-9 CODE '250' - Diabetes mellitus
    index = np.where(y_labels==target_ICD_code)[0] # index should be 95

    # for alpha_i, only need one row corresponding to ICD-9 of '250', alpha_i shape = (344,1)
    alpha_i = torch.transpose(alpha_i[index,:], 0 ,1)


    # we is essentially ksi_fc0, shape of (100,43325)

    lambda_i = np.zeros(all_vocab_size)
    # calculate lambda_250
    for j in range(all_vocab_size): # iterate over each word in the vocabulary
        # focus on jth word in the vocabulary, shape of (100,1)
        # skip word j, if the word is not in the Wiki document
        if(wiki_binary[index, j] == 0):
            continue
            
        we_temp = torch.unsqueeze(ksi_fc0[:, j], axis = -1)
 
    
        lambda_i[j] = ksi_fc2 @(alpha_i * we_temp * 1)

    # sort lambda in descending order
    sorted_index = np.argsort(-lambda_i)
    lambda_i = -np.sort(-lambda_i)

    # convert sorted_index to actual word
    result  = []

    for index in range(all_vocab_size):
        word_label = sorted_index[index]

        word_i = x_words_vocab_rev[word_label]

        # add pair to list
        result.append((word_i, lambda_i[index]))

    return result
      