from transformers import BertTokenizerFast, BertModel
from transformers import BertTokenizerFast, BertModel, BartTokenizer, BartForConditionalGeneration, AutoModel

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torch.nn import TransformerDecoder, TransformerDecoderLayer
from collections import OrderedDict
from itertools import groupby

from torch.nn import MultiheadAttention

import math
import numpy as np
from functools import lru_cache

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def masked_mean(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        return torch.mean(input, dim=dim)
    else:
        mask = mask.unsqueeze(-1)
        mask_input = input * mask
        sum_mask_input = mask_input.sum(dim=dim)
        for dim in range(mask.size(0)):
            sum_mask_input[dim] = sum_mask_input[dim] / mask[dim].sum()
        return sum_mask_input

class GCNGANModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels):
        super(GCNGANModel, self).__init__()

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.num_node_features = num_node_features
        self.graph_hidden_channels = graph_hidden_channels

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.text_proj_head = nn.Sequential(OrderedDict([
            ("text_hidden1", nn.Linear(self.ninp, self.nout)),
            ("ln2", nn.LayerNorm((self.nout)))
            ]))
        
        #For GCN:
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.graph_proj_head = nn.Sequential(OrderedDict([
            ("mol_hidden1", nn.Linear(self.graph_hidden_channels, self.nhid)),
            ("relu1", nn.ReLU(inplace=True)),
            ("mol_hidden2", nn.Linear(self.nhid, self.nhid)),
            ("relu2", nn.ReLU(inplace=True)),
            ("mol_hidden3", nn.Linear(self.nhid, self.nout)),
            ("ln1", nn.LayerNorm((self.nout)))
        ]))

        self.fc_sia = nn.Sequential(
            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.Tanh(),
        )

        self.other_params = list(self.parameters()) #get all but bert params
        
        self.discriminator = nn.Sequential(
          nn.Linear(self.nout, 512),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(512, 256),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(256, 1),
        )

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None, ot=False):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)
        text_x = text_encoder_output['pooler_output']
        text_x = self.text_proj_head(text_x)

        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        node_reps = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(node_reps, batch)  # [batch_size, graph_hidden_channels]
        x = self.graph_proj_head(x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        fc_text = norm(self.fc_sia(text_x))
        fc_mol = norm(self.fc_sia(x))

        return fc_text, fc_mol, text_x, x



class GCNGANQueryModel(nn.Module):
    def __init__(self, args):
        super(GCNGANQueryModel, self).__init__()

        self.ninp = args.ninp
        self.nhid = args.nhid
        self.nout = args.nout
        self.graph_hidden_channels = args.graph_hidden_channels
        self.num_node_features = args.num_node_features
        try:
            self.ot = 'ot' in args.loss
        except:
            self.ot = False
        self.interlingua_length = args.interlingua_length
        self.ma_type = args.ma_type
        self.query_update = args.query_update

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.text_proj_head = nn.Sequential(OrderedDict([
            ("text_hidden1", nn.Linear(self.ninp, self.nout)),
            ("ln2", nn.LayerNorm((self.nout)))
            ]))
        
        #For GCN:
        self.conv1 = GCNConv(self.num_node_features, self.graph_hidden_channels)
        self.conv2 = GCNConv(self.graph_hidden_channels, self.graph_hidden_channels)
        self.conv3 = GCNConv(self.graph_hidden_channels, self.graph_hidden_channels)
        self.graph_proj_head = nn.Sequential(OrderedDict([
            ("mol_hidden1", nn.Linear(self.graph_hidden_channels, self.nhid)),
            ("relu1", nn.ReLU(inplace=True)),
            ("mol_hidden2", nn.Linear(self.nhid, self.nhid)),
            ("relu2", nn.ReLU(inplace=True)),
            ("mol_hidden3", nn.Linear(self.nhid, self.nout)),
            ("ln1", nn.LayerNorm((self.nout)))
        ]))

        self.fc_sia = nn.Sequential(
            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.Tanh(),
        )


        self.interlingua_embedding = nn.Embedding(
                args.interlingua_length, self.nout, 0
        )
        if self.ma_type == 'query':
            self.cross_attention = MultiheadAttention(embed_dim=self.nout, num_heads=4, batch_first=True)

        if self.ma_type == 'query_glocal':
            self.global_interlingua_embedding = nn.Embedding(
                1, self.nout, 0
            )
            self.cross_attention = MultiheadAttention(embed_dim=self.nout, num_heads=4, batch_first=True)
            self.global_cross_attention = MultiheadAttention(embed_dim=self.nout, num_heads=4, batch_first=True)

            
        elif self.ma_type == 'decoder':
            self.cross_attention = nn.TransformerDecoderLayer(d_model=self.nout, nhead=4, batch_first=True)

        self.other_params = list(self.parameters()) #get all but bert params
        
        self.discriminator = nn.Sequential(
          nn.Linear(self.nout, 512),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(512, 256),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(256, 1),
        )

        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    @lru_cache(maxsize=1024)
    def translate(self, nodes, batch):
        """
        len_nodes * dim -> batch * len * dim
        """
        dim = nodes.size(-1)
        group_batch = groupby(batch, key=lambda x: x)
        padding = torch.ones((1, dim)).to(nodes.device)
        bsz = max(batch) + 1

        lens = [len(list(group)) for key, group in group_batch]
        max_len = max(lens)

        res = []
        mask = []
        pos = 0
        for i in range(bsz):
            res += [
                torch.cat(
                    (
                        nodes[pos : pos + lens[i]],
                        padding.repeat(((max_len - lens[i]), 1)),
                    ),
                    dim=0,
                ).unsqueeze(0)
            ]
            mask += [
                torch.cat(
                    (torch.zeros(lens[i]), torch.ones(max_len - lens[i]))
                ).unsqueeze(0)
            ]
            pos += lens[i]

        res = torch.cat(res, dim=0).to(nodes.device)
        mask = torch.cat(mask, dim=0).to(nodes.device).bool()
        return res, mask


    def get_text_rep(self, text, text_mask):
        batch_size = text.size(0)
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)
        token_reps = text_encoder_output['last_hidden_state']
        token_reps = self.text_proj_head(token_reps)
        query = self.interlingua_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.ma_type == 'query':
            token_reps, _ = self.cross_attention(query=query, key=token_reps, value=token_reps, key_padding_mask=~text_mask)
        text_x = torch.mean(token_reps, dim=1)
        text_x = text_x * torch.exp(self.temp)
        fc_text = norm(self.fc_sia(text_x))
        return fc_text


    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None, ot=False):
        batch_size = text.size(0)
        
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)
        token_reps = text_encoder_output['last_hidden_state']
        token_reps = self.text_proj_head(token_reps)

        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        node_reps = self.conv3(x, edge_index)
        node_reps = self.graph_proj_head(node_reps)
        node_reps, node_mask = self.translate(node_reps, batch)
        if self.query_update == 'cl':
            # query只用对比学习来更新，到这步的encoder用来计算对抗损失
            temp_text_x = masked_mean(token_reps, text_mask)
            temp_x = masked_mean(node_reps, ~node_mask)

        
        # query对齐
        query = self.interlingua_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.ma_type == 'query_glocal':
            token_reps, _ = self.cross_attention(query=query, key=token_reps, value=token_reps, key_padding_mask=~text_mask)
            atom_reps, _ = self.cross_attention(query=query, key=node_reps, value=node_reps, key_padding_mask=node_mask)

            global_query = self.global_interlingua_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            text_x, _ = self.global_cross_attention(query=global_query, key=token_reps, value=token_reps)
            x, _ = self.global_cross_attention(query=global_query, key=node_reps, value=node_reps)

            text_x = text_x.squeeze(dim=1)
            x = x.squeeze(dim=1)

            x = x * torch.exp(self.temp)
            text_x = text_x * torch.exp(self.temp)

            fc_text = norm(self.fc_sia(text_x))
            fc_mol = norm(self.fc_sia(x))
            
            ot_inp = None

            return fc_text, fc_mol, temp_text_x, temp_x, ot_inp


        if self.ma_type == 'query':
            token_reps, _ = self.cross_attention(query=query, key=token_reps, value=token_reps, key_padding_mask=~text_mask)
            atom_reps, _ = self.cross_attention(query=query, key=node_reps, value=node_reps, key_padding_mask=node_mask)
        elif self.ma_type == 'decoder':
            token_reps = self.cross_attention(tgt=query, memory=token_reps, memory_key_padding_mask=~text_mask)
            atom_reps = self.cross_attention(tgt=query, memory=node_reps, memory_key_padding_mask=node_mask)
        
        text_x = torch.mean(token_reps, dim=1)
        x = torch.mean(atom_reps, dim=1)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        fc_text = norm(self.fc_sia(text_x))
        fc_mol = norm(self.fc_sia(x))
        
        ot_inp = None

        return fc_text, fc_mol, temp_text_x, temp_x, (token_reps, atom_reps)