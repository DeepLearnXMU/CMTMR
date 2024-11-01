import os
import time
import csv
import sys
import math
import pickle
import numpy as np
import argparse
from datetime import datetime
import wandb
wandb.login()
import logging

import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from transformers.optimization import get_linear_schedule_with_warmup
from losses.cl_loss import contrastive_loss
from losses.triplet_loss import global_cos_loss, global_euc_loss, TripletLoss #, adaptive_triplet_loss
from losses.gan_loss import compute_gradient_penalty
from losses.dist_align_loss import kl_loss, ret_distill
from dataloaders import get_dataloader, GenerateData, get_graph_data, get_attention_graph_data, GenerateDataAttention, get_attention_dataloader, load_data
from utils.model_util import save_model, load_model, load_ckpt
from utils.seed_util import set_seed
from inference import test

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
triplet_loss = TripletLoss([0], margin=0.3)
def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def do_gcn_step(model, d, graph_batcher, optimizer=None, loss='cl', smoothing=0):
    batch, _ = d
    text_mask = batch['text']['attention_mask'].bool().to(device)
    text = batch['text']['input_ids'].to(device)
    graph_batch = graph_batcher(d[0]['molecule']['cid']).to(device)
    outputs = model(text, graph_batch, text_mask)
    text_out = outputs[0]
    chem_out = outputs[1]
    if 'cl' in loss:
        loss = contrastive_loss(text_out, chem_out, smoothing=smoothing).to(device)
    elif 'tl_euc' in loss:
        label = list(range(0, text_out.size(0)))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        loss = global_euc_loss(triplet_loss, torch.cat((text_out, chem_out)), label)[0]
    elif 'tl_cos' in loss:
        label = list(range(0, text_out.size(0)))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        loss = global_cos_loss(triplet_loss, torch.cat((text_out, chem_out)), label)[0]

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prefix = "train/"
    else:
        prefix = "valid/"

    loss_res = {prefix+"loss": loss.item(), prefix+"loss_contra": loss.item()}
    return loss_res


def do_gan_step(model, d, graph_batcher, optimizer=None, optimizer_D=None, loss='tl_euc_gan', current_epoch=None):
    loss_type = args.loss
    batch, _ = d
    text_mask = batch['text']['attention_mask'].bool().to(device)
    text = batch['text']['input_ids'].to(device)
    graph_batch = graph_batcher(d[0]['molecule']['cid']).to(device)
    outputs = model(text, graph_batch, text_mask)
    fc_text, fc_chem, text_out, chem_out = outputs[0], outputs[1], outputs[2], outputs[3]
    ################################################################
    # modal-level fusion
    ################################################################
    real_validity = model.discriminator(text_out.detach())
    fake_validity = model.discriminator(chem_out.detach())
    if optimizer:
        gradient_penalty = compute_gradient_penalty(model.discriminator, text_out.detach(), chem_out.detach())
    else:
        gradient_penalty = torch.tensor([0]).to(device) # 验证时不进行梯度惩罚的计算
    loss_cmD = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
    if optimizer_D:
        optimizer_D.zero_grad()
        loss_cmD.backward()
        optimizer_D.step()

    g_fake_validity = model.discriminator(chem_out)
    loss_cmG = -torch.mean(g_fake_validity)
    ################################################################
    # cross-modal retrieval
    ################################################################
    # if 'adap' in loss:
    #     adaptive_loss = adaptive_triplet_loss(fc_text, fc_chem) + adaptive_triplet_loss(fc_chem, fc_text)
    # else:
    #     adaptive_loss = torch.tensor([0]).to(device)

    if 'euc' in loss_type:
        label = list(range(0, fc_text.size(0)))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        contra_loss = global_euc_loss(triplet_loss, torch.cat((fc_text, fc_chem)), label)[0]
    elif 'cos' in loss_type:
        label = list(range(0, fc_text.size(0)))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).cuda().long()
        contra_loss = global_cos_loss(triplet_loss, torch.cat((fc_text, fc_chem)), label)[0]
    elif 'cl' in loss_type:
        contra_loss = contrastive_loss(fc_text, fc_chem).to(device)

    loss = args.contra_weight * contra_loss + args.gan_weight * loss_cmG
    
    ret_distill_res = ret_distill(args, fc_text, fc_chem)
    ret_distill_loss = torch.tensor([0]).to(device).float()
    if 'u2ucos' in loss_type:
        ret_distill_loss += args.u2u_weight * ret_distill_res['loss_u2u_cos']
        loss += args.u2u_weight * ret_distill_res['loss_u2u_cos']
    if 'u2ccos' in loss_type:
        ret_distill_loss += args.u2c_weight * ret_distill_res['loss_u2c_cos']
        loss += args.u2c_weight * ret_distill_res['loss_u2c_cos']
    if 'u2ueuc' in loss_type:
        ret_distill_loss += args.u2u_weight * ret_distill_res['loss_u2u_euc']
        loss += args.u2u_weight * ret_distill_res['loss_u2u_euc']
    if 'u2ceuc' in loss_type:
        ret_distill_loss += args.u2c_weight * ret_distill_res['loss_u2c_euc']
        loss += args.u2c_weight * ret_distill_res['loss_u2c_euc']
    if 'l2t' in loss_type:
        ret_distill_loss += args.u2c_weight * ret_distill_res['loss_2t']
        loss += args.u2c_weight * ret_distill_res['loss_2t']
    if 'l2m' in loss_type:
        ret_distill_loss += args.u2c_weight * ret_distill_res['loss_2m']
        loss += args.u2c_weight * ret_distill_res['loss_2m']
    if 'lt2m' in loss_type:
        ret_distill_loss += args.u2u_weight * ret_distill_res['loss_t2m']
        loss += args.u2u_weight * ret_distill_res['loss_t2m']
    if 'lm2t' in loss_type:
        ret_distill_loss += args.u2u_weight * ret_distill_res['loss_m2t']
        loss += args.u2u_weight * ret_distill_res['loss_m2t']
    

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prefix = "train/"
    else:
        prefix = "valid/"

    loss_res = {prefix+"loss": loss.item(), 
                prefix+"loss_contra": contra_loss.item(), 
                prefix+"loss_cmG": loss_cmG.item(), 
                prefix+"loss_cmD": loss_cmD.item(),
                prefix+"loss_ret_distill": ret_distill_loss.item(),
                prefix + 'loss_u2u_cos': ret_distill_res['loss_u2u_cos'].item(),
                prefix + 'loss_u2u_euc': ret_distill_res['loss_u2u_euc'].item(),
                prefix + 'loss_u2c_cos': ret_distill_res['loss_u2c_cos'].item(),
                prefix + 'loss_u2c_euc': ret_distill_res['loss_u2c_euc'].item(),
                }
                # prefix+"loss_adaptive": adaptive_loss.item(),
                # prefix+"loss_ot": loss_ot.item()}
    return loss_res


def main(args):
    set_seed(args.seed)
    MODEL = args.model

    output_path = os.path.join(args.home_dir, "output", args.model_name)
    args.output_path = output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'models'))
        os.mkdir(os.path.join(output_path, 'embeddings'))

    if args.wandb:
        
        wandb.init(project=args.wandb_project,
            name = args.wandb_name,
            config=args)

    now = datetime.now()  
    start_time = now.strftime("%m%d%H%M%S")  
    log_path = os.path.join(args.output_path, "{}_{}.log".format(args.model_name, start_time))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(), 
            logging.FileHandler(log_path), 
        ])
    logger = logging.getLogger("main")

    logger.info("loading data")
    train_data, valid_data, test_data, graph_tr, graph_val, graph_test, gd = load_data(args)
    logger.info("loading model")
    model = load_model(args)
    model.to(device)
    print(model)
    print(count_parameters(model))

    logger.info("loading optimizer")
    optimizer = optim.Adam([
                    {'params': model.other_params},
                    {'params': list(model.text_transformer_model.parameters()), 'lr': args.bert_lr}
                ], lr=args.lr)
    if 'gan' in args.loss:
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
    else:
        optimizer_D = None
    start_epoch = 0
    # if "checkpoint_last.pt" in os.listdir(os.path.join(output_path, "models")):
    #     last_path = os.path.join(output_path, "models", "checkpoint_last.pt")
    #     logger.info("continue training from {}".format(last_path))
    #     start_epoch = load_ckpt(last_path, model, optimizer=optimizer, strict=True)

    if len(args.train_from) > 0:
        logger.info("continue training from {}".format(args.train_from))
        checkpoint = torch.load(args.train_from)
        model.load_state_dict(checkpoint['model'], strict=True)

    num_training_steps = args.epochs * len(train_data) - args.num_warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_training_steps) 

    logger.info("start training")
    
    train_losses, val_losses = [], []
    current_step = 0

    for epoch in range(start_epoch, args.epochs):

        start_time = time.time()
        model.train()
        running_loss = 0.0
  
        for i, d in enumerate(train_data):
            current_step += 1
            if MODEL == "GCN":
                loss_res = do_gcn_step(model, d, graph_tr, optimizer, loss=args.loss, smoothing=args.label_smoothing)
            elif MODEL in ['GCN_GAN', 'GCN_QUERY', 'GCN_GAN_QUERY']:
                loss_res = do_gan_step(model, d, graph_tr, optimizer=optimizer, optimizer_D=optimizer_D, loss=args.loss, current_epoch=epoch)
            if args.wandb: wandb.log(loss_res, step=current_step)
            if args.wandb: wandb.log({"train/epoch": epoch}, step=current_step)
            scheduler.step()
            running_loss += loss_res['train/loss']
            if (i+1) % 100 == 0: 
                logger.info("{} batches trained. Avg loss:\t {} . Avg ms/step = {}".format(i + 1, running_loss / (1+i), 1000*(time.time()-start_time)/(i+1)))
            
        train_losses.append(running_loss / (i+1))
        logger.info("Epoch {} training loss:\t\t {} . Time = {} seconds.".format(epoch + 1, running_loss / (i+1), time.time()-start_time))

        logger.info("one epoch training finished, starting validation")
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            running_loss, running_contra_loss = 0.0, 0.0
            for i, d in enumerate(valid_data):
                if MODEL == "GCN":
                    loss_res = do_gcn_step(model, d, graph_val, loss=args.loss, smoothing=args.label_smoothing)
                elif MODEL in ['GCN_GAN', 'GCN_GAN_QUERY']:
                    loss_res = do_gan_step(model, d, graph_tr, optimizer=None, loss=args.loss, current_epoch=epoch)

                running_loss += loss_res['valid/loss']
                running_contra_loss += loss_res['valid/loss_contra']
                if (i+1) % 100 == 0: logger.info("{} batches eval. Avg loss:\t {}. Avg ms/step = {}".format(i + 1, running_loss / (1+i), 1000*(time.time()-start_time)/(i+1)))
            val_losses.append(running_loss / (i+1))
            val_contra = running_contra_loss / (i+1)

            logger.info("valid loss: {}, valid contra loss: {}".format(val_losses[-1], val_contra))
            if args.wandb: wandb.log({"valid/loss": val_losses[-1], "valid/loss_contra": val_contra}, step=current_step)
            min_loss = np.min(val_losses)
            if val_losses[-1] == min_loss:
                logger.info("valid loss drop, current min val loss is {}, save model...".format(min_loss))
                # save_model(epoch, model, optimizer, os.path.join(output_path, "models", 'weights_pretrained.{epoch:02d}-{min_loss:.2f}.pt'.format(epoch = epoch+1, min_loss = min_loss)))
                save_model(epoch, model, optimizer, os.path.join(output_path, "models", 'checkpoint_last.pt'))
        msg = "Epoch {},\t validation loss: {},\t Time = {} seconds.".format(epoch+1, running_loss / (i + 1),time.time()-start_time)
        logger.info(msg)

        if (epoch + 1) % 60 == 0:
            logger.info("60 epoch finished, starting testing")
            infer_res = test(args, model, logger, gd, (graph_tr, graph_val, graph_test))
            if args.wandb: wandb.log(infer_res, step=current_step)
            args.mol2text = 1
            infer_res = test(args, model, logger, gd, (graph_tr, graph_val, graph_test))
            if args.wandb: wandb.log(infer_res, step=current_step)
            

    logger.info("finish training")
    save_model(epoch, model, optimizer, os.path.join(output_path, "models", "final_weights."+str(epoch)+".pt"))
    save_model(epoch, model, optimizer, os.path.join(output_path, "models", 'checkpoint_last.pt'))
    if args.wandb: wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CMTMR')
    parser.add_argument('--data', type=str, default="/home/text2mol")
    parser.add_argument('--home-dir', type=str, default="/home/text2mol")
    parser.add_argument('--loss', type=str, default="cl", help='cl tl_euc tl_cos gan')
    parser.add_argument('--model', type=str, default='MLP', nargs='?',
                    help="model type from 'MLP, 'GCN', 'Attention'")
    parser.add_argument('--model-name', type=str, default='MLP')
    parser.add_argument('--mol_trunc_length', type=int, nargs='?', default=512)
    parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256)
    parser.add_argument('--num_warmup_steps', type=int, nargs='?', default=1000)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, nargs='?', default=1e-4)
    parser.add_argument('--bert_lr', type=float, nargs='?', default=3e-5)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--wandb-project', type=str, default="test")
    parser.add_argument('--wandb-name', type=str, default="test",)
    parser.add_argument('--encoder-embed-dim', type=int, default=300) # nout
    parser.add_argument('--encoder-attention-heads', type=int, default=4)
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder-normalize-before', type=bool, default=True)
    parser.add_argument('--encoder-ffn-embed-dim', type=int, default=1200)
    parser.add_argument('--ninp', type=int, default=768)
    parser.add_argument('--nhid', type=int, default=600)
    parser.add_argument('--nout', type=int, default=300)
    parser.add_argument('--num-node-features', type=int, default=300)
    parser.add_argument('--graph-hidden-channels', type=int, default=600)
    parser.add_argument('--modal-embedding', type=int, default=0)
    parser.add_argument('--interlingua-length', type=int, default=16)
    parser.add_argument('--interlingua-layer-num', type=int, default=3)
    parser.add_argument('--train-from', type=str, default='')
    parser.add_argument('--kl-type', type=str, default='atom_token2token')
    parser.add_argument('--contra-weight', type=float, default=10.0)
    parser.add_argument('--ret-distill-weight', type=float, default=0.0)
    parser.add_argument('--u2u-weight', type=float, default=0.0)
    parser.add_argument('--u2c-weight', type=float, default=0.0)
    parser.add_argument('--dis-align-weight', type=float, default=2e-3)
    parser.add_argument('--entr-align-weight', type=float, default=2e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--u2u-temp', type=float, default=1.0)
    parser.add_argument('--u2c-temp', type=float, default=1.0)
    parser.add_argument('--ma-type', type=str, default="query")
    parser.add_argument('--query-update', type=str, default="gan")
    parser.add_argument('--gan-weight', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mol2text', type=int, default=0)
    args = parser.parse_args()  
    print(args)
    main(args)
