#author: Niwhskal
#https://github.com/Niwhskal


import torch
import cfg
import torchvision.transforms.functional as F


def build_discriminator_loss(x_true, x_fake):

    # d_loss = -torch.mean(torch.log(torch.sigmoid(x_true)) + torch.log(1 - torch.sigmoid(x_fake)))
    d_loss = torch.mean((x_true - 1) ** 2 + x_fake ** 2)
    return d_loss

def build_dice_loss(x_t, x_o):
       
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat*tflat).sum()
    
    return 1. - torch.mean((2. * intersection + cfg.epsilon)/(iflat.sum() +tflat.sum()+ cfg.epsilon))

def build_l1_loss(x_t, x_o):
        
    return torch.mean(torch.abs(x_t - x_o))

def build_l1_loss_with_mask(x_t, x_o, mask):
    
    mask_ratio = 1. - mask.view(-1).sum() / torch.size(mask)
    l1 = torch.abs(x_t - x_o)
    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))

def build_perceptual_loss(x):        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_gram_matrix(x):

    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x):
        
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] *f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_vgg_loss(x):
        
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style

def build_gan_loss(x_pred):
    
    # gen_loss = -torch.mean(torch.log(torch.sigmoid(x_pred)))
    gen_loss = torch.mean((x_pred - 1) ** 2)
    
    return gen_loss

def build_generator_loss(out_g, out_d, out_vgg, labels):
        
    o_sk, o_t= out_g
    # o_dsk_pred, o_dt_pred = out_d
    o_dt_pred = out_d[0]
    o_vgg = out_vgg
    t_sk, t_t, t_b, t_f = labels

    l_sk_dice = build_dice_loss(t_sk, o_sk)
    # l_sk_gan = build_gan_loss(o_dsk_pred)
    l_sk_gan = 0
    l_sk = l_sk_dice + l_sk_gan

    l_t_l1 = build_l1_loss(t_t, o_t)
    l_t_gan = build_gan_loss(o_dt_pred)
    l_t_vgg_per, l_t_vgg_style = build_vgg_loss(o_vgg)
    l_t = 10 * l_t_l1 + l_t_gan + l_t_vgg_per + 500 * l_t_vgg_style
    
    l = l_t + l_sk
    return l, [l_sk_dice, l_sk_gan, l_t_l1, l_t_gan, l_t_vgg_per, l_t_vgg_style]
