import os
import argparse

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_livecell_loader, get_livecell_dataset
from torch.utils.data import Subset
torch.cuda.empty_cache()


######## for percentage od dataset , with MAE pretrain on livecell initialization
######## for percentage od dataset , with SAM pretrain initialization
def identity(raw):
        return raw

def do_unetr_training(data_path: str, save_root: str, cell_type: list, iterations: int, device, patch_shape=(224, 224)):
    os.makedirs(data_path, exist_ok=True)
    
    subset_indices = list(range(50))

    

    train_set = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=cell_type,
        download=True,
        boundaries=True,
        raw_transform = identity

    )

    sub_train_set = Subset(train_set, subset_indices)

    train_loader = torch_em.get_data_loader(sub_train_set, batch_size = 2)


    val_set = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=cell_type,
        download=True,
        boundaries=True

    )

    sub_val_set = Subset(val_set, subset_indices)

    val_loader = torch_em.get_data_loader(sub_val_set, batch_size = 1)

    

    out_channels = 2

    model = UNETR(
        img_size=args.img_size, backbone=args.backbone, encoder=args.encoder, out_channels = out_channels, use_sam_stats=args.use_sam_stats, encoder_checkpoint=args.checkpoint
    )  
    print("UNETR Model successfully created and encoder initialized from", args.checkpoint)
    
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name=f"unetr-source-livecell-{cell_type[0]}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=10,
        save_root=save_root,
        compile_model=False
    )

    trainer.fit(iterations)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on LiveCell dataset")
        do_unetr_training(
            data_path=args.inputs,
            save_root=args.save_root,
            cell_type=args.cell_type,
            iterations=args.iterations,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on LiveCELL dataset")
    parser.add_argument("-c", "--cell_type", nargs='+', default=["A172"],
                        help="Choice of cell-type for doing the training")
    parser.add_argument("-i", "--inputs", type=str, default="./livecell/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000, help="No. of iterations to run the training for")
    parser.add_argument("--checkpoint")
    parser.add_argument("--encoder", default="vit_l")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone")
    parser.add_argument("--use_sam_stats")
    args = parser.parse_args()
    main(args)

    
    #--train --cell_type "MCF7" --input /scratch/projects/cca/data/livecell/LiveCELL --save_root /scratch/users/menayat/models/UNETR_LiveCell_SAM_b_sub50 --iterations 10000 --checkpoint /scratch/users/menayat/models/SAM/sam_vit_b_01ec64.pth --encoder "vit_b" --backbone "sam" --use_sam_stats True
