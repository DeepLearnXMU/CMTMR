import torch
from torch_geometric.data import Dataset

from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer

class GINMatchDataset(Dataset):
    def __init__(self, root, args):
        super(GINMatchDataset, self).__init__(root)
        self.root = root
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.text_name_list = os.listdir(root+'text/')
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.data_type = args.data_type

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        data_aug = data_graph
        text_path = os.path.join(self.root, 'text', text_name)

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None
        
        if self.data_type == 1: #random sentence
            sts = text_list[0].split('.')
            remove_list = []
            for st in (sts):
                if len(st.split(' ')) < 5: 
                    remove_list.append(st)
            remove_list = sorted(remove_list, key=len, reverse=False)
            for r in remove_list:
                if len(sts) > 1:
                    sts.remove(r)
            text_index = random.randint(0, len(sts)-1)
            text, mask = self.tokenizer_text(sts[text_index])

        return data_aug, text.squeeze(0), mask.squeeze(0)#, index

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask