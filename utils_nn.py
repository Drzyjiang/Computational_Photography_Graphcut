# ksi functions
from utils_ksi import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# evaluate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


# implement a dataloader
def load_data(x_dataset, batch_size):
    batch_size = batch_size

    x_loader = DataLoader(x_dataset, batch_size = batch_size, shuffle = True, num_workers= 1)
    #val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers= 1)

    return x_loader


# evaluation RNN models
def eval_model(model, val_loader, device, code_group= None, y_labels = None):
    # set model in evaluation (inference) mode
    model.eval()

    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()

    # iterate over each batch
    num_batch = len(val_loader)
    for x, y in val_loader:
        # use gpu, if necessary
        x, y = x.to(device), y.to(device)
        
        print('remaining number of batches is: ', num_batch)

        # note that the return type is probability, not [0,1]
        y_pred_probas = model(x)
        #print('y_pred_probas: ', y_pred_probas)
        
        # convert probability to binary
        y_pred_binary = (y_pred_probas > 0.5).int()
        #print('y_pred_binary: ', y_pred_binary.max().to('cpu').numpy(), y_pred_binary.min().to('cpu').numpy())

        # concatenate predicated probabilties, predicated labels, and true labels from ALL batches
        y_score = torch.cat((y_score, y_pred_probas.detach().to('cpu')), dim = 0)
        y_pred = torch.cat((y_pred, y_pred_binary.detach().to('cpu')), dim = 0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim = 0)
        
        # update num_batch
        num_batch -= 1

    
    # After getting predicated values from ALL batches, evaluate precision, recall, f1, and roc_auc
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average = 'micro')
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro')

    
    # to calculate roc_auc, must REMOVE ICD-9 code with only one class
    y_pred_remove = y_pred.numpy()
    y_true_remove = y_true.numpy()
    
    column_to_remove = []
    
    for j in range(y_true_remove.shape[1]):
        #if(np.sum(y_true_remove[:,j]) == 0):
        if((y_true_remove[:,j] == 1).all() or (y_true_remove[:,j] == 0).all()):
            column_to_remove.append(j)
    
    
    print('There are', len(column_to_remove), 'ICD-9 code that only have ONE class.', \
        'The corresponding columns must be removed before evaluating ROC-AUC, because ROC-AUC in this case is UNDEFINED.')
    column_to_remove = np.array(column_to_remove)

    # remove those columns in both y_test and y_pred_proba_transpose
    y_true_remove = np.delete(y_true_remove, column_to_remove, axis = 1)
    y_pred_remove = np.delete(y_pred_remove, column_to_remove, axis = 1)

    roc_auc_micro = roc_auc_score(y_true_remove, y_pred_remove, average = 'micro', multi_class = 'ovr')
    roc_auc_macro = roc_auc_score(y_true_remove, y_pred_remove, average = 'macro', multi_class= 'ovr')
    
    
    # The paper asks to get macro AUC based on difference code frequency
    roc_auc_none = roc_auc_score(y_true_remove, y_pred_remove, average = None)
    auc_average = frequency_based_marcro_AUC(code_group, y_labels, roc_auc_none )
    print('The average macro AUC for each of the five frequency group is:')
    print(auc_average)
    
    return p_micro, p_macro, r_micro, r_macro, f1_micro, f1_macro, roc_auc_micro, roc_auc_macro
    

def frequency_based_marcro_AUC(code_group, y_labels, roc_auc_none ):
    # this function calculates macro AUC for five different groups of codes
    # the goal is to reproduce Figure 3 in the paper
    # each group has frequency range [1-10, 11-50, 51-100, 101-500, >500]
    
    # code_group: a list with size of 5, each element contains all ICD-9 code within the above frequency range 
    # y_labels: 1D array of string with size of 344, each string denotes a ICD-9 code
    
    if(code_group == None):
        return None
    
    assert len(code_group) == 5, 'code_frequency should be a list (of list), with len of 5.'
    
        
    auc_group =np.zeros(5)
    auc_count = np.zeros(5)
    
    # iterate over each column 
    

    
    for j in range(roc_auc_none.shape[0]):

        #print('roc_auc_none :', roc_auc_none[j])

        # find the acutal ICD-9 code
        code_name = y_labels[j]
        
        # find out current column j belong to which group
        if(code_name in code_group[0]): # code_name has occurrence between 1 and 10
            auc_group[0] += roc_auc_none[j]
            auc_count[0] += 1
        elif(code_name in code_group[1]): # code_name has occurrence between 11 and 50
            auc_group[1] += roc_auc_none[j]
            auc_count[1] += 1
        elif(code_name in code_group[2]): # code_name has occurrence between 51 and 100
            auc_group[2] += roc_auc_none[j]
            auc_count[2] += 1
        elif(code_name in code_group[3]): # code_name has occurrence between 101 and 500
            auc_group[3] += roc_auc_none[j]
            auc_count[3] += 1
        elif(code_name in code_group[4]): # code_name has occurrence between >500
            auc_group[4] += roc_auc_none[j]
            auc_count[4] += 1
    
    # calculate the average of macro AUC for each group
    auc_average = auc_group / auc_count
    
    return auc_average

# train and evaluation
def train(model, train_loader, val_loader, n_epochs, device, criterion, optimizer, code_group, y_labels):
    for epoch in range(n_epochs):
        # change the model to training status
        model.train()
        train_loss = 0
        
        train_loader_size = len(train_loader)
        print('The size of train_loader is:', train_loader_size)
        for x, y in train_loader:      
            y = y.to(torch.float32)
            print('Ground truth has number of positive of: ', torch.sum(y))
            # switch to GPU, if necessary
            x, y = x.to(device), y.to(device)
            
            
            
            outputs = model.forward(x)
            print('My model has number of positive of:', torch.sum(outputs >= 0.5).to('cpu').numpy())
            #print('outputs sum: ', torch.sum(outputs))
            
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            train_loader_size -= 1
            print('The remaining size of train_loader is:', train_loader_size)
        
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        
        p_micro, p_macro, r_micro, r_macro, f1_micro, f1_macro, roc_auc_micro, roc_auc_macro = eval_model(model, val_loader, device, code_group, y_labels)
        
        print('Epoch:', epoch+1, '\n',\
              'Validation Precision_micro:', p_micro,'\n', \
              'Validation Precision_macro:', p_macro,'\n', \
              'Recall_micro:', r_micro, '\n',\
              'Recall_macro:', r_macro, '\n',\
              'F1_micro:', f1_micro, '\n',\
              'F1_macro:', f1_macro, '\n', \
              'ROC-AUC_micro:', roc_auc_micro,'\n',\
              'ROC-AUC_macro:', roc_auc_macro)

# this model already integrates the KSI framework
class LR(nn.Module):
    def __init__(self, ksi_flag, num_codes,  embedding_dim, num_hidden, num_words,\
                 device, wiki_binary = None, total_vocabulary = None):
        super().__init__()


        self.ksi_flag = ksi_flag

        self.fc = nn.Linear(in_features = num_words, out_features = num_codes, bias = False) # the paper has no bias
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # objects for KSI if necessary
        if(ksi_flag == True):
            self.ksi_wiki_binary = wiki_binary
            self.ksi_total_vocabulary = total_vocabulary
            
            # self.ksi_fc0 is W_e
            self.ksi_fc0 = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
            self.ksi_fc1 = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
            self.ksi_fc2 = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)

    def forward(self, x):
        # x shape: (batch_size, vocabulary), where batch_size is number of clinical notes, num_words is length of longest note
        batch_size = x.shape[0]
        
        # pass the sequence through the fc layer.

        o = self.fc(x) # o.shape = (batch_size, num_code)

        # check if need to apply KSI
        if(self.ksi_flag in [True, 'ksi']):
            # use function subtask2 to get o2
            o2 = subtask2(x, self.ksi_wiki_binary, self.ksi_total_vocabulary, self.embedding_dim,\
                          self.ksi_fc0, self.ksi_fc1, self.ksi_fc2, self.device)
            
            o = o + o2

        # pass the o to sigmoid 
        probs = self.sigmoid(o)  # probs.shape = (batch_size,num_codes)
        
        
        return probs

# implement a dataloader
def load_data(x_dataset, batch_size):
    batch_size = batch_size

    x_loader = DataLoader(x_dataset, batch_size = batch_size, shuffle = True, num_workers= 1)
    #val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers= 1)

    return x_loader
    
    
    
# Build a RNN (many to many), based on long short-term memory
# this model already integrates the KSI framework
class RNN(nn.Module):
    def __init__(self, ksi_flag, num_codes, num_embeddings, embedding_dim, num_hidden, num_words,\
                 device, wiki_binary = None, total_vocabulary = None):
        super().__init__()


        self.ksi_flag = ksi_flag
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = self.embedding_dim)
        # the paper use num_layers = default = 1
        # input_size should be the embedding_dim
        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = num_hidden, num_layers = 1,\
                            batch_first = True, bidirectional = False)
        self.maxpool = nn.MaxPool1d(kernel_size = num_words)
        self.softmax = nn.Softmax(dim = 1)
        self.fc = nn.Linear(in_features = num_hidden, out_features = num_codes)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
        # objects for KSI if necessary
        if(ksi_flag in ['ksi', 'ksi_attention', 'ksi_intersection']):
            self.ksi_wiki_binary = wiki_binary
            self.ksi_total_vocabulary = total_vocabulary
            
            # self.ksi_fc0 is W_e
            self.ksi_fc0 = nn.Linear(in_features = self.ksi_total_vocabulary, out_features = embedding_dim, bias = False)
            self.ksi_fc1 = nn.Linear(in_features = embedding_dim, out_features = embedding_dim, bias = False)
            self.ksi_fc2 = nn.Linear(in_features = embedding_dim, out_features = 1, bias = True)

    def forward(self, x):
        # x shape: (batch_size, num_words), where batch_size is number of clinical notes, num_words is length of longest note
        batch_size = x.shape[0]
        
        #print('x ', x)

        # pass the sequence through the embedding layer. Each word is vector
        output = self.embedding(x) # x.shape = (batch_size, num_words, embedding_dim)
        #print('after embeding, x : ', x) # [32, 8232, 50]
        
        # specify (h0, c0)
        #h0 = torch.randn(1,batch_size, self.num_hidden).to(device)
        #c0 = torch.randn(1,batch_size, self.num_hidden).to(device)
        
        # pass the embeddings through the RNN layer
        # output.shape = (batch_size, num_words, num_hidden) # [32, 8232, 100]
        output, (final_hidden_state, final_cell_state) = self.rnn(output) ##, (h0, c0))

        # transpose from (batch_size, num_words, num_hidden) to (batch_size, num_hidden, num_words)
        output = output.permute(0, 2, 1)
        #print('output shape: ', output.shape)  # [32, 50, 8232]
        
        # pass the hidden state through the max pooling layer, kernel_size = W_length
        # shape from (batch_size, num_hidden, W_length) to (batch_size, num_hidden, 1), i.e., only apply on the last dimension
        output = self.maxpool(output)  # output.shape = (batch_size, num_hidden, 1)
        #print('after maxpool shape: ', output.shape) # [32, 50, 1]
        
        # remove the last dimension
        output = output[:,:,0]  # output.shape = (batch_size, num_hidden)
        #print('output shape: ', output.shape) # [32, 100]

        # apply the fc layer
        o = self.fc(output)  # output.shape = (batch_size, num_codes)
        #print('o shape: ', o.shape) # [32, 344]
        
        
        # check if need to apply KSI
        if(self.ksi_flag == 'ksi'):
            # use function subtask2 to get o2
            # subtask2
            o2 = subtask2(x, self.ksi_wiki_binary, self.ksi_total_vocabulary, self.embedding_dim,\
                          self.ksi_fc0, self.ksi_fc1, self.ksi_fc2, self.device)
            
            o = o + o2
        elif(self.ksi_flag =='ksi_attention'):
            # use function subtask2_attention to get o2
            # subtask2
            o2 = subtask2_attention(x, self.ksi_wiki_binary, self.ksi_total_vocabulary, self.embedding_dim,\
                          self.ksi_fc0, self.ksi_fc1, self.ksi_fc2, self.device)
            
            o = o + o2
        elif(self.ksi_flag =='ksi_intersection'):
            # use function subtask2_intersection to get o2
            # subtask2
            o2 = subtask2_intersection(x, self.ksi_wiki_binary, self.ksi_total_vocabulary, self.embedding_dim,\
                          self.ksi_fc0, self.ksi_fc1, self.ksi_fc2, self.device)
            
            o = o + o2

        # pass the o to sigmoid 
        probs = self.sigmoid(o)  # probs.shape = (batch_size,num_codes)

        return probs



