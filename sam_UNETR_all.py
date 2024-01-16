import os
import argparse

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_livecell_loader
from torch_em.data import ConcatDataset

def identity(raw):
        return raw


def do_unetr_training(args, data_path: str, save_root: str, iterations: int, device, patch_shape=(520, 704)):
    os.makedirs(data_path, exist_ok=True)
    train_set_A172 = get_livecell_dataset(
    path=data_path,
    split="train",
    patch_shape=patch_shape,
    cell_types=["A172"],                   
    download=True,
    boundaries=True,
    raw_transform = identity,
    n_samples=60

    )



    train_set_BT474 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["BT474"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_set_BV2 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["BV2"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_set_Huh7 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["Huh7"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )
    train_set_MCF7 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["MCF7"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_set_SHSY5Y = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["SHSY5Y"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_set_SkBr3 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["SkBr3"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_set_SKOV3 = get_livecell_dataset(
        path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=["SKOV3"],                   
        download=True,
        boundaries=True,
        raw_transform = identity,
        n_samples=60

        )

    train_dataset = ConcatDataset(train_set_A172, train_set_BT474, train_set_BV2, train_set_Huh7, train_set_MCF7, train_set_SHSY5Y, train_set_SkBr3, train_set_SKOV3)
    train_loader = torch_em.get_data_loader(train_dataset, batch_size = 2, num_workers=16,shuffle=True)



    val_set_A172 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["A172"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )



    val_set_BT474 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["BT474"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_set_BV2 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["BV2"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_set_Huh7 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["Huh7"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )
    val_set_MCF7 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["MCF7"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_set_SHSY5Y = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["SHSY5Y"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_set_SkBr3 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["SkBr3"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_set_SKOV3 = get_livecell_dataset(
        path=data_path,
        split="val",
        patch_shape=patch_shape,
        cell_types=["SKOV3"],                   
        download=True,
        boundaries=True,
        raw_transform = identity

        )

    val_dataset = ConcatDataset(val_set_A172, val_set_BT474, val_set_BV2, val_set_Huh7, val_set_MCF7, val_set_SHSY5Y, val_set_SkBr3, val_set_SKOV3)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size=1, num_workers=16,shuffle=True)

    # for x, y in train_loader:
    #     # ideally, the input images should be in range [0, 255]
    #     print(x.shape, y.shape)
    #     breakpoint()

    # breakpoint()

    model = UNETR(
        backbone=args.backbone, encoder=args.encoder, out_channels=2, use_sam_stats=True,
        encoder_checkpoint=args.checkpoint, final_activation="Sigmoid",
    )
    print("UNETR Model successfully created and encoder initialized from", args.checkpoint)
    
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name="unetr", #f"unetr-{cell_type[0]}"
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
            iterations=args.iterations,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on LiveCELL dataset")
    parser.add_argument("-i", "--inputs", type=str, default="./livecell/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000, help="No. of iterations to run the training for")
    parser.add_argument("--checkpoint")
    parser.add_argument("--encoder", default="vit_b")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone")
    args = parser.parse_args()
    main(args)

    #--input /scratch/projects/cca/data/livecell/LiveCELL --save_root /scratch/users/menayat/models/UNETR_LiveCell_MAE  checkpoints /scratch/users/menayat/models/MAE/checkpoint-0.pth
    #--train --cell_type "MCF7" --input /scratch/projects/cca/data/livecell/LiveCELL --save_root /scratch/users/menayat/models/UNETR_LiveCell_MAE --checkpoint /scratch/users/menayat/models/MAE/checkpoint-0.pth --backbone "mae"
    #--train --input /scratch-grete/projects/nim00007/data/LiveCELL/ --save_root /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/livecell_MCF7_10_vit_b --iterations 10000 --checkpoint /scratch-grete/usr/nimmahen/models/SAM/checkpoints/sam_vit_b_01ec64.pth --backbone "sam" --encoder "vit_b"
