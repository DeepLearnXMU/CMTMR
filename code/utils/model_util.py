import torch
import os
import json
import sys
from models.gcn import GCNModel
from models.gcn_gan import GCNGANModel, GCNGANQueryModel


def save_model(epoch, model, optimizer, file_save_path):
    opti = None
    if optimizer is not None:
        opti = optimizer.state_dict()

    torch.save(obj={
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opti,
    }, f=file_save_path)


def load_ckpt(resume_path, model, optimizer=None, strict=True):
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch



def load_model(args):
    MODEL = args.model
    if MODEL == "GCN":
        model = GCNModel(num_node_features=300, ninp = 768, nhid = 600, nout = 300, graph_hidden_channels = 600)
    elif MODEL == 'GCN_GAN':
        model = GCNGANModel(num_node_features=300, ninp = 768, nhid = 600, nout = 300, graph_hidden_channels = 600)
    elif MODEL == 'GCN_GAN_QUERY':
        model = GCNGANQueryModel(args)
    return model