import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from dual_model import *



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.CenterCrop(512),
                            transforms.ToTensor(),
                            normalize])

size=512
#
#trans = transforms.Compose([transforms.Resize(size),
#                            transforms.ToTensor(),
#                            normalize])


def calc_content_loss(out_features, t):
    return F.mse_loss(out_features, t)


def calc_style_loss(content_middle_features, style_middle_features):
    loss = 0
    for c, s in zip(content_middle_features, style_middle_features):
        c_mean, c_std = calc_mean_std(c)
        s_mean, s_std = calc_mean_std(s)
        loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)  #（batch,channel,1,1)
    return loss

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda()
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda()
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def MSE_Loss(x,y):
    mseloss=nn.MSELoss().cuda()
    return mseloss(x,y)

def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default='../content_test',
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default='../style_test',
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--attn_state_path', type=str, default='./model_state/attn_14_epoch.pth',
                        help='save directory for result and loss')
    parser.add_argument('--decoder_state_path', type=str, default='./model_state/decoder_14_epoch.pth',
                        help='save directory for result and loss')
    parser.add_argument('--vgg_state_path', type=str, default='./model_state/vgg_encoder_14_epoch.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"


    # set model
#    model = Model()
    
    attn=CoAttention(channel=512).cuda()
    decoder = Decoder().cuda()
    vgg_encoder = VGGEncoder().cuda()
    
    
    
    
    if args.attn_state_path is not None:
#        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
        kwargs={'map_location':lambda storage, loc: storage}
#        attn=load_GPUS(attn,args.attn_state_path,kwargs)#如果多卡训练的时候，忘记用torch.save(module.state_dict())
#        decoder=load_GPUS(decoder,args.decoder_state_path,kwargs)
#        vgg_encoder=load_GPUS(vgg_encoder,args.vgg_encoder_state_path,kwargs)
        
        
        attn.load_state_dict(torch.load(args.attn_state_path,map_location=lambda storage, loc: storage))#如果多卡训练的时候，忘记用torch.save(module.state_dict())
        decoder.load_state_dict(torch.load(args.decoder_state_path,map_location=lambda storage, loc: storage))
        vgg_encoder.load_state_dict(torch.load(args.vgg_state_path,map_location=lambda storage, loc: storage))
        
        
        
    


    content_paths = []
    style_paths = []
    content_names = []
    style_names = []

    for root,_,fnames in os.walk(args.content):
        for name in fnames:
            content_names.append(name)
            content_paths.append(os.path.join(args.content,name))
    
    for root,_,fnames in os.walk(args.style):
        for name in fnames:
            style_names.append(name)
            style_paths.append(os.path.join(args.style,name))
    


    save_path=args.attn_state_path.split('/')
    save_path = save_path[0]+'/image_new_new'+save_path[2].split('.')[0]+'/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    print(style_paths)
    alpha=1.0
    
    c1=0.0
    c3=0.0
    c6=0.0
    c8=0.0
    s1=0.0
    s3=0.0
    s6=0.0
    s8=0.0
    i=0.0
    mse1=0.0
    mse2=0.0
    p1='./423.jpg'
    p2='./candy.jpg'
    p3='./423_stylized_candy.jpg'
    p4='./423_stylized_candy1.jpg'
    p5='./423_stylized_candy_stylized_423_stylized_candy1.jpg'
    p6='./423_stylized_candy_stylized_423_stylized_candy11.jpg'
    
    
    c=Image.open(p1)
    c_tensor = trans(c).unsqueeze(0).cuda()
    
    c=Image.open(p2)
    s_tensor = trans(c).unsqueeze(0).cuda()
    
    c=Image.open(p3)
    c1s2 = trans(c).unsqueeze(0).cuda()
    
    c=Image.open(p4)
    c2s1 = trans(c).unsqueeze(0).cuda()
    
    c=Image.open(p5)
    c1s1 = trans(c).unsqueeze(0).cuda()
    
    c=Image.open(p6)
    c2s2 = trans(c).unsqueeze(0).cuda()
    
    content_features = vgg_encoder(c_tensor, output_last_feature=True)
    style_features = vgg_encoder(s_tensor, output_last_feature=True)

    output_features_1 = vgg_encoder(c1s2, output_last_feature=True)
    output_features_2 = vgg_encoder(c2s1, output_last_feature=True)
    
    loss_c_1 = calc_content_loss(output_features_1, content_features)
    loss_c_6 = calc_content_loss(output_features_2, style_features)



    output_middle_features_1 = vgg_encoder(c1s2, output_last_feature=False)
    output_middle_features_2 = vgg_encoder(c2s1, output_last_feature=False)
    
    style_middle_features_1 = vgg_encoder(c_tensor, output_last_feature=False)
    style_middle_features_2 = vgg_encoder(s_tensor, output_last_feature=False)
    
    
    loss_s_1 = calc_style_loss(output_middle_features_1, style_middle_features_2)
    loss_s_6 = calc_style_loss(output_middle_features_2, style_middle_features_1)
    
#            print(loss_c_1.item(),loss_c_6.item(),loss_s_1.item(),loss_s_6.item())
    


    content_features_c1s2 = vgg_encoder(c1s2, output_last_feature=True)
    style_features_c2s1 = vgg_encoder(c2s1, output_last_feature=True)

    
    
    output_features_1 = vgg_encoder(c1s1, output_last_feature=True)
    output_features_2 = vgg_encoder(c2s2, output_last_feature=True)
    
    loss_c_3 = calc_content_loss(output_features_1, content_features)

    loss_c_8 = calc_content_loss(output_features_2, style_features)
    
    
    
    output_middle_features_1 = vgg_encoder(c1s1, output_last_feature=False)
    output_middle_features_2 = vgg_encoder(c2s2, output_last_feature=False)
    
    
    
    loss_s_3 = calc_style_loss(output_middle_features_1, style_middle_features_1)
    loss_s_8 = calc_style_loss(output_middle_features_2, style_middle_features_2)
    
#            print(loss_c_3.item(),loss_c_8.item(),loss_s_3.item(),loss_s_8.item())
    mse_1=MSE_Loss(c_tensor,c1s1)
    mse_2=MSE_Loss(s_tensor,c2s2)
    c1=loss_c_1.item()+c1
    c3=loss_c_3.item()+c3
    c6=loss_c_6.item()+c6
    c8=loss_c_8.item()+c8
    s1=loss_s_1.item()+s1
    s3=loss_s_3.item()+s3
    s6=loss_s_6.item()+s6
    s8=loss_s_8.item()+s8
    mse1=mse1+mse_1
    mse2=mse1+mse_2
    i=i+1

    
    
    
    print(i)
    print(c1/i,c3/i,c6/i,c8/i)
    print(s1/i,s3/i,s6/i,s8/i)
    print(mse1/i,mse2/i)
        

                
                
                
                


if __name__ == '__main__':
    main()
