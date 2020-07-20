#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""

import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.model import RawNet
from loss import GE2ELoss
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')






def train(model,dataloader_train,epoch,ce_loss,g2e_loss,optimizer,device):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        
    
        features = torch.cat((sample_batched[0])).float()
        labels = torch.cat((sample_batched[1]))
        
        features, labels = features.to(device),labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits,embeddings = model(features)
        embeddings = embeddings.reshape(10,10,128) ### [langs,utts,embed_dim]
        #### CE loss
        
        CE = ce_loss(pred_logits,labels)
        G2E = g2e_loss(embeddings)
        loss = CE+G2E
        #### CE loss
        
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        #train_acc_list.append(accuracy)
        #if i_batch%10==0:
        #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
    


def validation(model,dataloader_val,epoch,ce_loss,g2e_loss,optimizer,device):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.cat((sample_batched[0])).float()
            labels = torch.cat((sample_batched[1]))
        
            features, labels = features.to(device),labels.to(device)
            pred_logits,embeddings = model(features)
            embeddings = embeddings.reshape(10,10,128) ### [langs,utts,embed_dim]
            #### CE loss
            
            CE = ce_loss(pred_logits,labels)
            G2E = g2e_loss(embeddings)
            loss = CE+G2E
            val_loss_list.append(loss.item())
            #train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total vlidation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
        
        model_save_path = os.path.join('save_model', 'best_check_point_'+str(epoch)+'_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)

def main(args):
    
    ### Data related
    dataset_train = SpeechDataGenerator(manifest=args.training_filepath,mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 
    
    dataset_val = SpeechDataGenerator(manifest=args.validation_filepath,mode='train')
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 
    
    
    #dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
    #dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 
    
    ## Model related
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = RawNet(args.input_dim, args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
    ce_loss = nn.CrossEntropyLoss()
    g2e_loss  =GE2ELoss(device)
    for epoch in range(args.num_epochs):
        train(model,dataloader_train,epoch,ce_loss,g2e_loss,optimizer,device)
        validation(model,dataloader_val,epoch,ce_loss,g2e_loss,optimizer,device)
        



if __name__ == '__main__':
    
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-training_filepath',type=str,default='meta/training.txt')
    parser.add_argument('-testing_filepath',type=str, default='meta/testing.txt')
    parser.add_argument('-validation_filepath',type=str, default='meta/validation.txt')
    
    parser.add_argument('-input_dim', action="store_true", default=257)
    parser.add_argument('-num_classes', action="store_true", default=8)
    parser.add_argument('-lamda_val', action="store_true", default=0.1)
    parser.add_argument('-batch_size', action="store_true", default=10)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    args = parser.parse_args()
    
    main(args)
   