import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from dual_model import *

import time

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#trans = transforms.Compose([transforms.CenterCrop(512),
#                            transforms.ToTensor(),
#                            normalize])

trans = transforms.Compose([transforms.CenterCrop(512),
                            transforms.ToTensor(),
                            normalize])

size=512
#
#trans = transforms.Compose([transforms.Resize(size),
#                            transforms.ToTensor(),
#                            normalize])



def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda()
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda()
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res



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
    parser.add_argument('--content', '-c', type=str, default='./video_image',
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default='./video_image_style',
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--attn_state_path', type=str, default='./model_state/attn_14_epoch.pth',
                        help='save directory for result and loss')
    parser.add_argument('--decoder_state_path', type=str, default='./model_state/decoder_14_epoch.pth',
                        help='save directory for result and loss')
    parser.add_argument('--vgg_state_path', type=str, default='./model_state/vgg_encoder_14_epoch.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
    save_path = save_path[0]+'/image_video_image_resize'+save_path[2].split('.')[0]+'/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    
    alpha=1.0
    for i,c in enumerate(content_paths):
        print(c)
        c = Image.open(c)
        c_tensor = trans(c).unsqueeze(0).cuda()
        for j,s in enumerate(style_paths):
            
#            start=time.time()
            s = Image.open(s)
            
            s_tensor = trans(s).unsqueeze(0).cuda()
            with torch.no_grad():
                
                
                content_features = vgg_encoder(c_tensor, output_last_feature=True)
#                
                style_features = vgg_encoder(s_tensor, output_last_feature=True)
#                
#                
#                content_features, style_features=attn(content_features,style_features)
#                
                t_1 = adain(content_features, style_features)
                t_2=adain(style_features,content_features)
#                
#                
#                start=time.time()
#                
                c1s2 = decoder(t_1)
#                end=time.time()
#                
#                
                t_1 = alpha * t_1 + (1 - alpha) * content_features
                t_2=alpha*t_2+(1-alpha)*style_features
#                
                c2s1 = decoder(t_2)
#
#                
    
                content_features_c1s2 = vgg_encoder(c1s2, output_last_feature=True)
                style_features_c2s1 = vgg_encoder(c2s1, output_last_feature=True)
                
                content_features_c1s2, style_features_c2s1=attn(content_features_c1s2,style_features_c2s1)
                
                t_1_c1s1 = adain(content_features_c1s2, style_features_c2s1)
                t_2_c2s2=adain(style_features_c2s1,content_features_c1s2)
                
                
                
                t_1_c1s1 = alpha * t_1_c1s1 + (1 - alpha) * content_features_c1s2
                t_2_c2s2=alpha*t_2_c2s2+(1-alpha)*style_features_c2s1
                
                
                c1s1 = decoder(t_1_c1s1)
                c2s2 = decoder(t_2_c2s2)
                
#                print(end-start)
                
                
                
                
                
                
                
                c_denorm = denorm(c_tensor)
                s_denorm = denorm(s_tensor)
                out_1denorm = denorm(c1s2)
                out_2denorm = denorm(c2s1)
                c1s1_denorm=denorm(c1s1)
                c2s2_denorm=denorm(c2s2)
#            res = torch.cat([c_denorm, s_denorm, out_1denorm], dim=0)
            res = torch.cat([c_denorm, s_denorm, out_1denorm,out_2denorm,c1s1_denorm,c2s2_denorm], dim=0)
            
        
            res = res.to('cpu')
            

    
            # if args.output_name is None:
            c_name = os.path.splitext(os.path.basename(content_names[i]))[0]

            s_name = os.path.splitext(os.path.basename(style_names[j]))[0]

#            args.output_name = '{}_{}_{}'.format(args.alpha,c_name,s_name)7
            args.output_name = '{}'.format(c_name)
    
            res_1=c_denorm.to('cpu')
#            res_2=s_denorm.to('cpu')
#            res_3=out_1denorm.to('cpu')
#            res_4=out_2denorm.to('cpu')
#            res_5=c1s1_denorm.to('cpu')
#            res_6=c2s2_denorm.to('cpu')
#    
#    
#            save_image(c_denorm, save_path+'{}.jpg'.format(c_name))
#            save_image(s_denorm, save_path+'{}.jpg'.format(s_name))
#            save_image(out_denorm, save_path+'{}.jpg'.format(args.output_name), nrow=6)
            
#            
#            
#            save_image(res, save_path+'{}_pair.jpg'.format(args.output_name), nrow=6)
            save_image(res_1, save_path+'{}.jpg'.format(args.output_name), nrow=1)
#            save_image(res_2, save_path+'{}_pair.jpg'.format(args.output_name), nrow=1)
#            save_image(res_3, save_path+'{}_pair.jpg'.format(args.output_name), nrow=1)
#            save_image(res_4, save_path+'{}_pair.jpg'.format(args.output_name), nrow=1)
#            save_image(res_5, save_path+'{}_pair.jpg'.format(args.output_name), nrow=1)
#            save_image(res_6, save_path+'{}_pair.jpg'.format(args.output_name), nrow=3)
            
            
            
            
            
            
    
#            o = Image.open(save_path+'{}_pair.jpg'.format(args.output_name))
#            s = s.resize((i // 4 for i in c.size))
#            
#            box = (2*o.width // 3, o.height - s.height)
#            o.paste(s, box)
#            o.save(save_path+'{}_style_transfer_demo.jpg'.format(args.output_name), quality=95)
#            print('result saved into files starting with {}'.format(args.output_name))
            


if __name__ == '__main__':
    main()
