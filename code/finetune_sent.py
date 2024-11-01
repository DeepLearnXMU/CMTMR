
import numpy as np
import csv
import shutil
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import os.path as osp
import zipfile
from torch.utils.data import RandomSampler
import random
from transformers import BertTokenizerFast, BertModel

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Data, Batch
import torch_geometric

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
from utils.model_util import save_model, load_model, load_ckpt
from utils.seed_util import set_seed
from inference import test

from tqdm import tqdm, trange

class GINMatchDataset(Dataset):
    def __init__(self, split, args):
        super(GINMatchDataset, self).__init__()
        self.split = split
        self.text_max_len = 256
        self.graph_name_list = os.listdir('/home/data/PCdes_sent/{}/graph/'.format(split))
        self.text_name_list = os.listdir('/home/data/PCdes_sent/{}/text'.format(split))
        self.tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")

    def __len__(self):
        return len(self.text_name_list)

    def __getitem__(self, index):
        text_name = self.text_name_list[index]
        text_path = os.path.join('/home/data/PCdes_sent/{}/text'.format(self.split), text_name)
        cid = int(text_name.split(".")[0])
        graph_name = "{}.pt".format(cid)
        graph_path = os.path.join('/home/data/PCdes_sent/{}/graph/'.format(self.split), graph_name)
        data_graph = torch.load(graph_path)
        data_aug = data_graph

        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 500:
                break
        
        text = mask = None

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
                                   add_special_tokens=True,
                                   max_length=256,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


class GINSentDataset(Dataset):
    def __init__(self, split, args):
        super(GINSentDataset, self).__init__()
        self.split = split
        self.text_max_len = 256
        self.text_name_list = os.listdir('/home/data/PCdes_sent/{}/text'.format(self.split))
        self.tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
        
        self.all_text = []
        self.all_mask = []
        self.cor = []
        cnt = 0

        print("gin sent dataset len:", len(self.text_name_list))
        for text_name in self.text_name_list:
            text_path = os.path.join('/home/data/PCdes_sent/{}/text'.format(self.split), text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line.strip('\n')
                text_list.append(line)
                if count > 500:
                    break

            sts = text_list[0].split('.')
            self.cor.append(cnt)
            for st in sts:
                if len(st.split(' ')) < 5:
                    continue
                text, mask = self.tokenizer_text(st)
                self.all_text.append(text)
                self.all_mask.append(mask)
                cnt+=1
        self.cor.append(cnt)
        print("gin sent dataset len(cor): ", len(self.cor), self.cor[0], self.cor[-1])
        np.save('/home/data/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/cor.npy', self.cor)
        
    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, index):
        text = self.all_text[index]
        mask = self.all_mask[index]
        return text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   max_length=256,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
        attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
        return input_ids, attention_mask


def CalSent(model, dataloader, device, args): 
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in tqdm(dataloader):
            text, mask = batch
            text = text.cuda()
            mask = mask.cuda()
            mask = mask.bool()
            text_rep = model.get_text_rep(text, mask)
            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)
    return text_rep_total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
triplet_loss = TripletLoss([0], margin=0.3)
def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def do_gan_step(model, batch, optimizer=None, optimizer_D=None, loss_type='tl_euc_gan', current_epoch=None, distill_temp=1.0):
    graph_batch, text, text_mask = batch
    graph_batch.to(device)
    text = text.cuda()
    text_mask = text_mask.cuda()
    text_mask = text_mask.bool()

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


def Eval(model, dataloader, device, args, sent_dataloader=None):
    test_results = {}
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for batch in tqdm(dataloader):
            graph_batch, text, text_mask = batch
            graph_batch.to(device)
            text = text.cuda()
            text_mask = text_mask.cuda()
            text_mask = text_mask.bool()

            outputs = model(text, graph_batch, text_mask)
            text_rep, graph_rep= outputs[0], outputs[1]

            scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1==torch.arange(argm1.shape[0]).cuda()).int()).item()
            acc2 += sum((argm2==torch.arange(argm2.shape[0]).cuda()).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)
    test_results['test/acc1'] = acc1/allcnt
    test_results['test/acc2'] = acc2/allcnt
    print("acc1: {}, acc2: {}".format(acc1/allcnt, acc2/allcnt))

    


    assert sent_dataloader is not None
    text_rep_total = CalSent(model, sent_dataloader, device, args)
    print("finish cal sent")
    graph_rep = graph_rep_total.cpu()
    text_rep = text_rep_total.cpu()
    cor = np.load('/home/data/CMTMR/output/finetune_sent_q32sota_pcdes/embeddings/cor.npy')

    graph_len = graph_rep.shape[0]
    text_len = text_rep.shape[0]

    score1 = torch.zeros(graph_len, graph_len)
    score2 = torch.zeros(graph_len, graph_len)

    for i in trange(graph_len):
        score = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
        for j in range(graph_len):
            total = 0
            for k in range(cor[j], cor[j+1]):
                total+=(score[k]/(cor[j+1]-cor[j]))
            score1[i,j] = total
            #score1[i,j] = sum(score[cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    rec1 = []
    for i in trange(graph_len):
        a,idx = torch.sort(score1[:,i])
        for j in range(graph_len):
            if idx[-1-j]==i:
                rec1.append(j)
                break
    test_results['test/rec@20_1'] = sum( (np.array(rec1)<20).astype(int) ) / graph_len

    score_tmp = torch.zeros(text_len, graph_len)
    for i in trange(text_len):
        score_tmp[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
    score_tmp = torch.t(score_tmp)

    for i in trange(graph_len):
        for j in range(graph_len):
            total = 0
            for k in range(cor[j], cor[j+1]):
                total+=(score_tmp[i][k]/(cor[j+1]-cor[j]))
            score2[i,j] = total
            #score2[i,j] = sum(score_tmp[i][cor[j]:cor[j+1]])/(cor[j+1]-cor[j])
    score2 = torch.t(score2)

    rec2 = []
    for i in trange(graph_len):
        a,idx = torch.sort(score2[:,i])
        for j in range(graph_len):
            if idx[-1-j]==i:
                rec2.append(j)
                break
    test_results['test/rec@20_2'] = sum( (np.array(rec2)<20).astype(int) ) / graph_len

    return test_results

def main(args):
    if args.wandb:
        wandb.init(project=args.wandb_project,
            name = args.wandb_name,
            config=args)
        
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
      
    now = datetime.now()  # 获取当前日期和时间
    start_time = now.strftime("%m%d%H%M%S")  # 格式化日期和时间
    log_path = os.path.join(args.output_path, "{}_{}.log".format(args.model_name, start_time))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(), # 将日志信息输出到控制台
            logging.FileHandler(log_path), # 将日志信息写入文件
        ])
    logger = logging.getLogger("main")
    logger.info("loading data")
    dataset_name = 'PCdes_sent'
    text_trunc_length = 256
    TrainSet = GINMatchDataset("train", args)
    DevSet = GINMatchDataset('valid', args)
    TestSet = GINMatchDataset('test', args)
    train_sampler = RandomSampler(TrainSet)
    train_data = torch_geometric.loader.DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=1, pin_memory=True, drop_last=True)
    valid_data = torch_geometric.loader.DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=1, pin_memory=True, drop_last=True)
    test_data = torch_geometric.loader.DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=1, pin_memory=True, drop_last=False)#True

    logger.info("loading model")
    model = load_model(args)
    model.to(device)
    print(model)
    print(count_parameters(model))

    logger.info("continue training from {}".format(args.train_from))
    checkpoint = torch.load(args.train_from)
    model.load_state_dict(checkpoint['model'], strict=False)

    logger.info("loading optimizer")
    optimizer = optim.Adam([
                    {'params': model.other_params},
                    {'params': list(model.text_transformer_model.parameters()), 'lr': args.bert_lr}
                ], lr=args.lr)

    if 'gan' in args.loss:
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
    else:
        optimizer_D = None

    SentSet = GINSentDataset("test", args)
    sent_dataloader = torch_geometric.loader.DataLoader(SentSet, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=1, pin_memory=True, drop_last=False)#True
    print("finish loading sent dataloader")

    logger.info("start training")
    train_losses, val_losses = [], []
    current_step = 0
    for epoch in range(0, args.epochs):
      start_time = time.time()
      model.train()
      running_loss = 0.0

      for i, batch in tqdm(enumerate(train_data)):
          current_step += 1
          loss_res = do_gan_step(model, batch, optimizer=optimizer, optimizer_D=optimizer_D, loss_type=args.loss, current_epoch=epoch)
          if args.wandb: wandb.log(loss_res, step=current_step)
          if args.wandb: wandb.log({"train/epoch": epoch}, step=current_step)
          running_loss += loss_res['train/loss']
          if (i+1) % 100 == 0: 
              logger.info("{} batches trained. Avg loss:\t {} . Avg ms/step = {}".format(i + 1, running_loss / (1+i), 1000*(time.time()-start_time)/(i+1)))
            
      train_losses.append(running_loss / (i+1))
      logger.info("Epoch {} training loss:\t\t {} . Time = {} seconds.".format(epoch + 1, running_loss / (i+1), time.time()-start_time))

    #   logger.info("one epoch training finished, starting validation")
    #   model.eval()
    #   with torch.no_grad():
    #       start_time = time.time()
    #       running_loss, running_contra_loss = 0.0, 0.0
    #       for batch in tqdm(valid_data):
    #           loss_res = do_gan_step(model, batch, optimizer=None, optimizer_D=None, loss_type=args.loss, current_epoch=epoch)
    #           running_loss += loss_res['valid/loss']
    #           running_contra_loss += loss_res['valid/loss_contra']
    #           if (i+1) % 100 == 0: logger.info("{} batches eval. Avg loss:\t {}. Avg ms/step = {}".format(i + 1, running_loss / (1+i), 1000*(time.time()-start_time)/(i+1)))
    #       val_losses.append(running_loss / (i+1))
    #       val_contra = running_contra_loss / (i+1)

    #       logger.info("valid loss: {}, valid contra loss: {}".format(val_losses[-1], val_contra))
    #       if args.wandb: wandb.log({"valid/loss": val_losses[-1], "valid/loss_contra": val_contra}, step=current_step)
    #       min_loss = np.min(val_losses)
    #       if val_losses[-1] == min_loss:
    #           logger.info("valid loss drop, current min val loss is {}, save model...".format(min_loss))
    #           save_model(epoch, model, optimizer, os.path.join(output_path, "models", 'checkpoint_last.pt'))
    #   msg = "Epoch {},\t validation loss: {},\t Time = {} seconds.".format(epoch+1, running_loss / (i + 1),time.time()-start_time)
    #   logger.info(msg)

      if (epoch + 1) % 10 == 0:
          logger.info("10 epoch finished, starting testing")
          infer_res = Eval(model, test_data, "cuda", args, sent_dataloader)
          logger.info("finishing testing, test res: {}".format(str(infer_res)))
          if args.wandb: wandb.log(infer_res, step=current_step)

    logger.info("finish training")
    save_model(epoch, model, optimizer, os.path.join(output_path, "models", "final_weights."+str(epoch)+".pt"))
    save_model(epoch, model, optimizer, os.path.join(output_path, "models", 'checkpoint_last.pt'))
    if args.wandb: wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CMTMR')
    parser.add_argument('--home-dir', type=str, default="/home/CMTMR")
    parser.add_argument('--loss', type=str, default="cl", help='cl tl_euc tl_cos gan')
    parser.add_argument('--model', type=str, default='MLP', nargs='?',
                    help="model type from 'MLP, 'GCN', 'Attention'")
    parser.add_argument('--model-name', type=str, default='MLP')
    parser.add_argument('--mol_trunc_length', type=int, nargs='?', default=512)
    parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256)
    parser.add_argument('--num_warmup_steps', type=int, nargs='?', default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, nargs='?', default=5e-5)
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
    parser.add_argument('--interlingua-length', type=int, default=32)
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
    parser.add_argument('--if-test', type=int, default=2)
    parser.add_argument('--datatype', type=int, default=1) # 0 for para, 1 for sent
    args = parser.parse_args()  
    main(args)