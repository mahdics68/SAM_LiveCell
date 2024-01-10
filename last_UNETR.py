import os
import argparse

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_livecell_loader

def identity(raw):
        return raw


def do_unetr_training(args, data_path: str, save_root: str, cell_type: list, iterations: int, device, patch_shape=(520, 704)):
    os.makedirs(data_path, exist_ok=True)
    train_loader = get_livecell_loader(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=cell_type,
        download=True,
        boundaries=True,
        raw_transform = identity,
        num_workers=16,
        shuffle=True,
        n_samples=args.n_samples
    )

    val_loader = get_livecell_loader(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        cell_types=cell_type,
        download=True,
        boundaries=True,
        raw_transform=identity,
        num_workers=16,
        shuffle=True
    )

    model = UNETR(
        backbone=args.backbone, encoder=args.encoder, out_channels=2, use_sam_stats=args.use_sam_stats,
        encoder_checkpoint=args.checkpoint, final_activation="Sigmoid",
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
        compile_model=False,
        mixed_precision=True
    )

    trainer.fit(iterations)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on LiveCell dataset")
        do_unetr_training(
            args,
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
    parser.add_argument("--use_sam_stats", action="store_true")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)

    #--input /scratch/projects/cca/data/livecell/LiveCELL --save_root /scratch/users/menayat/models/UNETR_LiveCell_MAE  checkpoints /scratch/users/menayat/models/MAE/checkpoint-0.pth
    #--train --cell_type "MCF7" --input /scratch/projects/cca/data/livecell/LiveCELL --save_root /scratch/users/menayat/models/UNETR_LiveCell_MAE --checkpoint /scratch/users/menayat/models/MAE/checkpoint-0.pth --backbone "mae"
    #--train --cell_type "MCF7" --input /scratch-grete/projects/nim00007/data/LiveCELL/ --save_root /scratch-grete/usr/nimmahen/models/UNETR/checkpoints/ --iterations 10000--checkpoint /scratch-grete/usr/nimmahen/models/SAM/checkpoints/sam_vit_l_0b3195.pth --backbone "sam" --use_sam_stats True
