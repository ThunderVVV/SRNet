# Training script for the SRNet. Refer README for instructions.
# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage import io
from model import Generator, Discriminator, Vgg19
from torchvision import models, transforms, datasets
from loss import build_generator_loss, build_discriminator_loss
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def custom_collate(batch):
    
    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []
    
    w_sum = 0

    for item in batch:
        
        t_b= item[4]
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)
        
    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)
    
    for item in batch:
   
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = item


        i_t = resize(i_t, to_scale, preserve_range=True)
        i_s = resize(i_s, to_scale, preserve_range=True)
        t_sk = np.expand_dims(resize(t_sk, to_scale, preserve_range=True), axis = -1) 
        t_t = resize(t_t, to_scale, preserve_range=True)
        t_b = resize(t_b, to_scale, preserve_range=True)  
        t_f = resize(t_f, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)


        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1)) 

        i_t_batch.append(i_t) 
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t) 
        t_b_batch.append(t_b) 
        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.) 
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.) 
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.) 
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)    

      
    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]

def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)  # 0
    
    train_name = get_train_name()  # 用时间命名
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    
    train_data = datagen_srnet(cfg)  # Dataset
    
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)
    # Dataloader

    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
        
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
    
    print_log('training start.', content_color = PrintColor['yellow'])
        
    G = Generator(in_channels = 3).cuda()
    # Dsk = Discriminator(in_channels=4).cuda()
    Dt = Discriminator(in_channels=6).cuda()
        
    vgg_features = Vgg19().cuda()    
        
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    # Dsk_solver = torch.optim.Adam(Dsk.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    Dt_solver = torch.optim.Adam(Dt.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_solver, milestones=[20, 30], gamma=0.1)    
    # dsk_scheduler = torch.optim.lr_scheduler.MultiStepLR(Dsk_solver, milestones=[20, 30], gamma=0.1)    
    dt_scheduler = torch.optim.lr_scheduler.MultiStepLR(Dt_solver, milestones=[20, 30], gamma=0.1)

    try:
    
      checkpoint = torch.load(cfg.ckpt_path)
      G.load_state_dict(checkpoint['generator'])
    #   Dsk.load_state_dict(checkpoint['discriminatorsk'])
      Dt.load_state_dict(checkpoint['discriminatort'])
      G_solver.load_state_dict(checkpoint['g_optimizer'])
    #   Dsk_solver.load_state_dict(checkpoint['dsk_optimizer'])
      Dt_solver.load_state_dict(checkpoint['dt_optimizer'])
      
      g_scheduler.load_state_dict(checkpoint['g_scheduler'])
    #   dsk_scheduler.load_state_dict(checkpoint['dsk_scheduler'])
      dt_scheduler.load_state_dict(checkpoint['dt_scheduler'])

      print('Resuming after loading...')

    except FileNotFoundError:

      print('checkpoint not found')
      pass  

    requires_grad(G, False)
    # requires_grad(Dsk, True)
    requires_grad(Dt, True)        
    
    trainiter = iter(train_data)
    example_iter = iter(example_loader)
    
    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    writer = SummaryWriter(log_dir=cfg.TRAIN_TENSORBOARD_DIR)

    for step in tqdm(range(cfg.max_iter)):
        
        # Dsk_solver.zero_grad()
        Dt_solver.zero_grad()
        
        if ((step+1) % cfg.save_ckpt_interval == 0):  # 每1000次保存一次
            
            torch.save(
                {
                    'generator': G.state_dict(),
                    # 'discriminatorsk': Dsk.state_dict(),
                    'discriminatort': Dt.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    # 'dsk_optimizer': Dsk_solver.state_dict(),
                    'dt_optimizer': Dt_solver.state_dict(),
                    'g_scheduler' : g_scheduler.state_dict(),
                    # 'dsk_scheduler':dsk_scheduler.state_dict(),
                    'dt_scheduler':dt_scheduler.state_dict(),
                },
                cfg.checkpoint_savedir+f'train_step-{step+1}.model',
            )
        
        # 处理每个epoch最后一次迭代
        try:

          i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = trainiter.next()

        except StopIteration:
          g_scheduler.step()
        #   dsk_scheduler.step()
          dt_scheduler.step()

          trainiter = iter(train_data)
          i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = trainiter.next()
                
        i_t = i_t.cuda()
        i_s = i_s.cuda()
        t_sk = t_sk.cuda()
        t_t = t_t.cuda()
        t_b = t_b.cuda()
        t_f = t_f.cuda()
        mask_t = mask_t.cuda()
                
        #inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3])) #Adding dim info
        
        # padding
        # print("t_sk:{}".format(t_sk.shape))
        # print("o_sk:{}".format(o_sk.shape))
        # print("t_t:{}".format(t_t.shape))
        # print("o_t:{}".format(o_t.shape))

        o_sk = K(o_sk)
        o_t = K(o_t)

        # print(torch.sum(torch.isnan(o_t)))
        
        i_dsk_true = torch.cat((t_sk, i_s), dim=1)
        i_dsk_pred = torch.cat((o_sk, i_s), dim=1)

        i_dt_true = torch.cat((t_t, i_s), dim=1)
        i_dt_pred = torch.cat((o_t, i_s), dim=1)

        # o_dsk_true = Dsk(i_dsk_true)
        # o_dsk_pred = Dsk(i_dsk_pred)

        o_dt_true = Dt(i_dt_true)
        o_dt_pred = Dt(i_dt_pred)

        # dsk_loss = build_discriminator_loss(o_dsk_true, o_dsk_pred)
        dt_loss = build_discriminator_loss(o_dt_true, o_dt_pred)
       
        # dsk_loss.backward()
        dt_loss.backward()

        # Dsk_solver.step()
        Dt_solver.step()        
        
        if ((step+1) % 1 == 0):  # FIXME 控制判别器每迭代几次后再迭代生成器 
            
            requires_grad(G, True)
            # requires_grad(Dsk, False)
            requires_grad(Dt, False)
            
            G_solver.zero_grad()
            
            o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            
            o_sk = K(o_sk)
            o_t = K(o_t)

            i_dsk_pred = torch.cat((o_sk, i_s), dim=1)

            i_dt_pred = torch.cat((o_t, i_s), dim=1)

            # o_dsk_pred = Dsk(i_dsk_pred)

            o_dt_pred = Dt(i_dt_pred)

            i_vgg = torch.cat((t_t, o_t), dim = 0)

            out_vgg = vgg_features(i_vgg)
            
            out_g = [o_sk, o_t]
        
            out_d = [o_dt_pred]
        
            g_loss, detail = build_generator_loss(out_g, out_d, out_vgg, labels)    
                
            g_loss.backward()
            
            G_solver.step()
                        
            requires_grad(G, False)

            # requires_grad(Dsk, True)
            requires_grad(Dt, True)
            
        if ((step+1) % cfg.write_log_interval == 0):
            
            # print('Iter: {}/{} | g_loss: {} | dsk_loss: {} | dt_loss: {}'.format(step+1, cfg.max_iter, g_loss.item(), dsk_loss.item(), dt_loss.item()))
            print('Iter: {}/{} | g_loss: {} | dt_loss: {}'.format(step+1, cfg.max_iter, g_loss.item(), dt_loss.item()))
            print('Iter: {}/{} | l_sk_dice: {} | l_sk_gan: {} | l_t_l1: {} | l_t_gan: {} | l_t_vgg_per: {} | l_t_vgg_style: {}'.format(
                step + 1,
                cfg.max_iter,
                detail[0].item(),
                detail[1],
                detail[2].item(),
                detail[3].item(),
                detail[4].item(),
                detail[5].item()
            ))

            writer.add_scalar('train/g_loss', g_loss.item(), step+1)
            # writer.add_scalar('train/dsk_loss', dsk_loss.item(), step+1)
            writer.add_scalar('train/dt_loss', dt_loss.item(), step+1)
            writer.add_scalar('train/l_sk_dice', detail[0].item(), step+1)
            writer.add_scalar('train/l_sk_gan', detail[1], step+1)
            writer.add_scalar('train/l_t_l1', detail[2].item(), step+1)
            writer.add_scalar('train/l_t_gan', detail[3].item(), step+1)
            writer.add_scalar('train/l_t_vgg_per', detail[4].item(), step + 1)
            writer.add_scalar('train/l_t_vgg_style', detail[5].item(), step + 1)
            writer.add_scalar('lr', G_solver.param_groups[0]['lr'], step + 1)

        if ((step+1) % cfg.gen_example_interval == 0):  # 每1000次执行一次
            G.eval()
            
            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))

            with torch.no_grad():

                try:

                  inp = example_iter.next()
                
                except StopIteration:

                  example_iter = iter(example_loader)
                  inp = example_iter.next()
                
                i_t = inp[0].cuda()
                i_s = inp[1].cuda()
                name = str(inp[2][0])
                
                o_sk, o_t = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

                o_sk = o_sk.squeeze(0).to('cpu')
                o_t = o_t.squeeze(0).to('cpu')
                
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                
                o_sk = F.to_pil_image(o_sk)
                o_t = F.to_pil_image((o_t + 1)/2)
                
                o_sk.save(os.path.join(savedir, name + 'o_sk.png'))
                o_t.save(os.path.join(savedir, name + 'o_t.png'))
            G.train()
    writer.close()
                
if __name__ == '__main__':
    main()
