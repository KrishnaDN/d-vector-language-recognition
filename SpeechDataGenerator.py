#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
from utils import utils
from itertools import compress
import random

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode):
        """
        Read the textfile and get the paths
        """
        self.mode=mode
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        class_id = self.labels[idx]
        ### select M random files
        get_ids = [el==class_id for el in self.labels]
        get_all_files = list(compress(self.audio_links, get_ids))
        selected_files = random.sample(get_all_files, 10)
        specs = []
        labels_list = []
        for audio_filepath in selected_files:
            spec = utils.load_data(audio_link,mode=self.mode)
            specs.append(spec)
            labels_list.append(class_id)
        feats = np.asarray(specs)
        label_arr = np.asarray(labels_list)
        
        sample = {'features': torch.from_numpy(np.ascontiguousarray(feats)), 'labels': torch.from_numpy(np.ascontiguousarray(label_arr))}
        return sample
        
    
