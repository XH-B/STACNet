import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from dual_model import *
import torch.nn as nn

def MSE_Loss(x,y):
    mseloss=nn.MSELoss().cuda()
    return mseloss(x,y)

def calc_content_loss(out_features, t):
    return F.mse_loss(out_features, t)


def calc_style_loss(content_middle_features, style_middle_features):
    loss = 0
    for c, s in zip(content_middle_features, style_middle_features):
        c_mean, c_std = calc_mean_std(c)
        s_mean, s_std = calc_mean_std(s)
        loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)  #（batch,channel,1,1)
    return loss

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=12,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=900,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--train_content_dir', type=str, default='../content',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', type=str, default='../style',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', type=str, default='content',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='style',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default=None,
                        help='model state path to load for reuse')

    args = parser.parse_args()

    print(args.save_dir)
    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'
    image_dir = f'{args.save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
    if not os.path.exists(model_state_dir):
        os.mkdir(model_state_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)




    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

    
    
    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')
    

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(args.train_content_dir, args.train_style_dir)
    test_dataset = PreprocessDataset(args.test_content_dir, args.test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_iter = iter(test_loader)

    device_ids=[0,1,2]
#    Re_encoder=nn.DataParallel(ReEncoder(),device_ids).cuda()
    vgg_encoder=nn.DataParallel(VGGEncoder(),device_ids).cuda()
    attn=nn.DataParallel(CoAttention(channel=512),device_ids).cuda()
    decoder=nn.DataParallel(Decoder(),device_ids).cuda()
    vggattn=nn.DataParallel(VGGAttn(),device_ids).cuda()
    D_img=Dimg().cuda()
    
    
    
    
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse))
        
        
#    optimizer_Re_encoder = Adam(Re_encoder.parameters(), lr=args.learning_rate)    
    optimizer_decoder = Adam(decoder.parameters(), lr=args.learning_rate)
    optimizer_attn = Adam(attn.parameters(), lr=args.learning_rate)
    optimizer_vggattn = Adam(filter(lambda p: p.requires_grad, vggattn.parameters()), lr=args.learning_rate)
    optimizer_D_img = Adam(D_img.parameters(), lr=args.learning_rate)

    # start training
    loss_list_1 = []
    loss_list_2=[]
    loss_list_D_img=[]
    lam=10.0
#    print(list(vggattn.parameters()))
    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.cuda()
            style = style.cuda()
            t_1=vggattn(content,style,output_last_feature=True)
            t_2=vggattn(style,content,output_last_feature=True)

            c1s2 = decoder(t_1)
            output_features_1 = vgg_encoder(images=c1s2, output_last_feature=True)
            output_middle_features_1 = vgg_encoder(images=c1s2, output_last_feature=False)
            style_middle_features_1 = vgg_encoder(images=style, output_last_feature=False)
            loss_c_1 = calc_content_loss(output_features_1, t_1)
            loss_s_1 = calc_style_loss(output_middle_features_1, style_middle_features_1)
    
            c2s1 = decoder(t_2)
            output_features_2 = vgg_encoder(images=c2s1, output_last_feature=True)
            output_middle_features_2 = vgg_encoder(images=c2s1, output_last_feature=False)
            style_middle_features_2 = vgg_encoder(images=content , output_last_feature=False)
            loss_c_2 = calc_content_loss(output_features_2, t_2)
            loss_s_2 = calc_style_loss(output_middle_features_2, style_middle_features_2)
        
            D_content= D_img(content.to('cuda:1'))
            D_style=D_img(style.to('cuda:1'))
            D_c1s2=D_img(c1s2.to('cuda:1'))
            D_c2s1=D_img(c2s1.to('cuda:1'))
            
            

            
            
            D_loss=MSE_Loss(D_content,fake_label)+MSE_Loss(D_style,fake_label)+MSE_Loss(D_c1s2,real_label)+MSE_Loss(D_c2s1,real_label)
            
            loss =  loss_c_1 +lam * loss_s_1 + loss_c_2 + lam * loss_s_2+ 0.01*D_loss.to('cuda:0')
            
            
            loss_list_1.append(loss.sum().item())
            
            optimizer_vggattn.zero_grad()
            optimizer_decoder.zero_grad()
            loss.sum().backward(retain_graph=True)
            optimizer_decoder.step()
            optimizer_vggattn.step()
#            
            
            t_1_c1s1=vggattn(c1s2,c2s1,output_last_feature=True)
            t_2_c2s2=vggattn(c2s1,c1s2,output_last_feature=True)
            c1s1 = decoder(t_1_c1s1)
            c2s2 = decoder(t_2_c2s2)
#            
#            
#            
            output_features_c1s1 = vgg_encoder(images=c1s1, output_last_feature=True)
            output_middle_features_c1s1 = vgg_encoder(images=c1s1, output_last_feature=False)
            style_middle_features_c1s1 = vgg_encoder(images=content, output_last_feature=False)
            c_old=vgg_encoder(images=content,output_last_feature=True)
            loss_c_c1s1 = calc_content_loss(output_features_c1s1, c_old) #与原图比较
            loss_s_c1s1 = calc_style_loss(output_middle_features_c1s1, style_middle_features_c1s1)
    
            
            
            
            output_features_c2s2 = vgg_encoder(images=c2s2,output_last_feature=True)
            s_old=vgg_encoder(images=style,output_last_feature=True)
            output_middle_features_c2s2 = vgg_encoder(images=c2s2, output_last_feature=False)
            style_middle_features_c2s2 = vgg_encoder(images=style, output_last_feature=False)
            loss_c_c2s2 = calc_content_loss(output_features_c2s2, s_old)#与原图比较
            loss_s_c2s2 = calc_style_loss(output_middle_features_c2s2, style_middle_features_c2s2)
            
            
            mse_c1s1=MSE_Loss(content,c1s1)
            mse_c2s2=MSE_Loss(style,c2s2)

            loss =loss_c_c1s1+lam*loss_s_c1s1+loss_c_c2s2+lam*loss_s_c2s2+10*mse_c1s1+10*mse_c2s2
            loss_list_1.append(loss.sum().item())


            optimizer_vggattn.zero_grad()
            optimizer_decoder.zero_grad()
            loss.sum().backward(retain_graph=True)
            optimizer_decoder.step()
            optimizer_vggattn.step()
#            
            for g_index in range(g_steps):
                optimizer_D_img.zero_grad()
                D_loss=MSE_Loss(D_content,real_label)+MSE_Loss(D_style,real_label)+MSE_Loss(D_c1s2,fake_label)+MSE_Loss(D_c2s1,fake_label)
                D_loss.backward()
                optimizer_D_img.step()

            print(f'[{e}/total {args.epoch} epoch],[{i} /'
                  f'total {round(iters/args.batch_size)} iteration]: {loss.sum().item()}')

    
            if i % args.snapshot_interval == 0:
           
                content = denorm(content)
                style = denorm(style)
                c1s2=denorm(c1s2)
                c2s1=denorm(c2s1)
                c1s1=denorm(c1s1)
                c2s2=denorm(c2s2)
                res = torch.cat([content, style, c1s2,c2s1,c1s1,c2s2], dim=0)
                
                res = res.to('cpu')
                save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=args.batch_size)        
#        torch.save(attn.state_dict(), f'{model_state_dir}/attn_{e}_epoch.pth')
        torch.save(vgg_encoder.state_dict(), f'{model_state_dir}/vgg_encoder_{e}_epoch.pth')
        torch.save(decoder.state_dict(), f'{model_state_dir}/decoder_{e}_epoch.pth')
        torch.save(D_img.state_dict(), f'{model_state_dir}/D_img_{e}_epoch.pth')
        torch.save(vggattn.state_dict(), f'{model_state_dir}/vggattn{e}_epoch.pth')

        with open(f'{loss_dir}/loss_log.txt', 'w') as f:
            for l in loss_list_1:
                f.write(f'{l}\n')
        
    # plt.plot(range(len(loss_list)), loss_list)
    # plt.xlabel('iteration')
    # plt.ylabel('loss')
    # plt.title('train loss')
    # plt.savefig(f'{loss_dir}/train_loss.png')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
