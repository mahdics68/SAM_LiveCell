import torch
import glob
import torch_em
from torch_em.model import UNETR
import imageio.v2 as imageio
from torch_em.util.prediction import predict_with_halo
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse



def identity(raw):
        return raw


def test_unetr(args, model_weights: str, pred_dir:str):
       
    cell_types =  ["[MCF7]"] #"[A172]", "[BT474]", "[BV2]", "[Huh7]", "[MCF7]", "[SHSY5Y]", "[SkBr3]", "[SKOV3]"
    for i in cell_types:
                

                
                test_images = os.path.join("/scratch-grete/projects/nim00007/data/LiveCELL/images/livecell_test_images/", i+"*")

                model = UNETR(
                    backbone=args.backbone, encoder=args.encoder, out_channels=2, use_sam_stats=args.use_sam_stats,
                    final_activation="Sigmoid",
                    )
        
   
                model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu'))["model_state"])

                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

                model.to(device)
                model.eval()

                with torch.no_grad():
                    
                    for img_path in glob.glob(test_images):
                        
                        filename = os.path.split(img_path)[-1] 
                        # print(i, " -- ", filename)

                        inputs = imageio.imread(img_path)
                        #print('before std', inputs.shape)

                        #inputs = torch_em.transform.raw.standardize(inputs) ###for pretrained : no need #### for scratch: need it
                        
                        predictions = predict_with_halo(inputs, model, gpu_ids=[device], block_shape=(384,384), halo = (68,68)) #, preprocess=None
                       # for pretrained: preprocess = none , 
                       # for scratch : remove it (or preprocess=standardize  which is default )
                        

                        segmentation = predictions[0,:,:]
                        boundaries = predictions[1,:,:]
                        
                        
                        
                        
                        seg_dir = os.path.join(args.base_dir, pred_dir, "segmentation")
                        os.makedirs(seg_dir, exist_ok=True)
                        imageio.imwrite(os.path.join(seg_dir, filename), segmentation)

                        bd_dir = os.path.join(args.base_dir, pred_dir, "boundaries")
                        os.makedirs(bd_dir, exist_ok=True)
                        imageio.imwrite(os.path.join(bd_dir, filename), boundaries)
                    
                        

def main(args):
    test_unetr(
        args,
        model_weights=args.model_weights,
        pred_dir=args.pred_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="vit_b")
    parser.add_argument("--backbone")
    parser.add_argument("--use_sam_stats", action="store_true")
    parser.add_argument("--model_weights")
    parser.add_argument("--base_dir", default="/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/")
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    main(args)
                        
#/scratch/users/menayat/models/UNETR_LiveCell_MAE_SC/checkpoints/unetr-source-livecell-MCF7/best.pt
# --encoder "vit_b" --backbone "sam" --model_weights /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/livecell_MCF7_10_vit_b_Btch2/checkpoints/unetr-MCF7/best.pt --pred_dir livecell_MCF7_10_vit_b_Btch2
