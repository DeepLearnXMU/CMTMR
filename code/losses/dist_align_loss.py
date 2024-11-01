import torch.nn as nn
import torch
import torch.nn.functional as F
kl = nn.KLDivLoss(reduction='batchmean')

def kl_loss(text_decoder_dist, mm_decoder_dist, mask):
    # of p from q(p_y text)
    mm_decoder_dist = mm_decoder_dist.view(-1, mm_decoder_dist.size(-1)).contiguous()
    text_decoder_dist = text_decoder_dist.view(-1, text_decoder_dist.size(-1)).contiguous()
        
    logp_x = F.log_softmax(mm_decoder_dist, dim=-1)
    p_y = F.softmax(text_decoder_dist, dim=-1)

    klloss = F.kl_div(logp_x, p_y, reduction='none').sum(-1)
    klloss.masked_fill(mask.view(-1), 0.0)
    klloss = klloss.mean()
    return klloss

def u2u_euc_loss(text_features, mol_features, temp=1.0):
    mask = torch.zeros(size=(text_features.size(0), )).bool().to(text_features.device)
    text2text = euclidean_dist(text_features, text_features)
    mol2mol = euclidean_dist(mol_features, mol_features)
    lt2m = kl_loss(mol2mol / temp, text2text / temp, mask)
    lm2t = kl_loss(text2text / temp, mol2mol / temp, mask)
    return lt2m, lm2t

def u2c_euc_loss(text_features, mol_features, temp=1.0):
    mask = torch.zeros(size=(text_features.size(0),)).bool().to(text_features.device)
    text2text = euclidean_dist(text_features, text_features)
    mol2text = euclidean_dist(mol_features, text_features)
    mol2mol = euclidean_dist(mol_features, mol_features)
    text2mol = euclidean_dist(text_features, mol_features) 
    text2mol_loss = kl_loss(mol2mol.detach() / temp, text2mol / temp, mask)
    mol2text_loss = kl_loss(text2text.detach() / temp, mol2text / temp, mask)
    res = text2mol_loss + mol2text_loss
    return text2mol_loss, mol2text_loss

def cos_dist(x, y):
    batch_size = x.size(0)
    cos_sim_matrix = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
    dist = 0.5 * (cos_sim_matrix + 1)
    return dist

def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  # import pdb; pdb.set_trace()
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist

def ret_distill(args, text_features, mol_features, mix_epoch=None, temp=1.0):
    u2u_temp = args.u2u_temp
    u2c_temp = args.u2c_temp

    mask = torch.zeros(size=(text_features.size(0), )).bool().to(text_features.device)
    text2text = cos_dist(text_features, text_features)
    mol2text = cos_dist(mol_features, text_features)
    
    mol2mol = cos_dist(mol_features, mol_features)
    text2mol = cos_dist(text_features, mol_features) 

    text2mol_loss = kl_loss(mol2mol.detach() / u2c_temp, text2mol / u2c_temp, mask)
    mol2text_loss = kl_loss(text2text.detach() / u2c_temp, mol2text / u2c_temp, mask)
    modalities_simi = kl_loss(mol2mol / u2u_temp, text2text / u2u_temp, mask) + kl_loss(text2text / u2u_temp, mol2mol / u2u_temp, mask)

    lt2m, lm2t = u2u_euc_loss(text_features, mol_features, u2u_temp)
    l2m, l2t = u2c_euc_loss(text_features, mol_features, u2c_temp)
    return {"loss_u2u_cos": modalities_simi,
    "loss_u2u_euc": lt2m + lm2t,
    "loss_u2c_cos": text2mol_loss + mol2text_loss,
    "loss_u2c_euc": l2t + l2m,
    "loss_t2m": lt2m, "loss_m2t": lm2t, "loss_2m":l2m, "loss_2t": l2t}

