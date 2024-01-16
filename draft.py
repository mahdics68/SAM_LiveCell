
import torch
import torch_em
from torch_em.data.datasets import get_livecell_loader, get_livecell_dataset
from torch_em.data import ConcatDataset
from torch.utils.data import Subset
#from micro_sam.training import identity
import os
import argparse



# # cell_type = ["MCF7"] #"A172", "BT474","BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"
# # data_loader_train = get_livecell_loader(
# #     path=path,
# #     split="train",
# #     patch_shape=patch_shape,
# #     batch_size=2,
# #     cell_types=cell_type,
# #     download=True,
# #     boundaries=True,

# # )

# # dataset_size = len(data_loader_train.dataset)
# # print("Dataset size:", dataset_size)
# def identity(raw):
#         return raw

# data_path = '/scratch/projects/cca/data/livecell/LiveCELL/'
# patch_shape = (224,224)
# cell_type = ["MCF7"] #"A172", "BT474","BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"
# train_set = get_livecell_dataset(path=data_path,
#         split="train",
#         patch_shape=patch_shape,
#         cell_types=cell_type,
#         download=True,
#         boundaries=True,
#         raw_transform = identity

#     )

# subset_indices = list(range(50))
# subset_trainset = Subset(train_set, subset_indices)


# train_loader = torch_em.get_data_loader(subset_trainset, batch_size = 2 )

# for batch in train_loader:
#     # Extract the raw image from the batch (assuming raw is the first element in the batch)
#     raw_image = batch[0]#[0]  # Assuming the first index corresponds to the raw image and we're extracting the first image in the batch

#     # Check the value range of the raw image
#     min_value = torch.min(raw_image).item()
#     max_value = torch.max(raw_image).item()

#     print(f"Min value: {min_value}, Max value: {max_value}")

#     # Break the loop after processing the first batch (and first image in that batch)
#     break

 
# #print(len(subset_trainset))
# # print(train_set.dtype) 

# import imageio as io 

# img = io.imread('/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/livecell_MCF7_10_vit_b_Btch2/segmentation/MCF7_Phase_H4_1_00d00h00m_3.tif')
# print(img.shape)


# def identity(raw):
#      return raw

# data_path = '/scratch-grete/projects/nim00007/data/LiveCELL/'
# patch_shape = (520, 704)
# cell_type = None

# train_loader = get_livecell_loader(
#         path=data_path,
#         split="train",
#         patch_shape=patch_shape,
#         batch_size=2,
#         cell_types=cell_type,
#         download=True,
#         boundaries=True,
#         #raw_transform = identity,
#         num_workers=16,
#         shuffle=True,
#         #n_samples=50
#     )

# print(len(train_loader.dataset))


# def test_unetr(args, pred_dir:str):

    
#     seg_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/", args.pred_dir , "segmentation/")
#     os.makedirs(seg_dir, exist_ok=True)
#     #imageio.imwrite(os.path.join(seg_dir, filename), segmentation)

# def main(args):
#     test_unetr(
#         args,
#         pred_dir=args.pred_dir
#         )


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pred_dir")
#     args = parser.parse_args()
    
#     main(args)




# def test_unetr(args, pred_dir):
#     seg_dir = os.path.join(args.base_dir, pred_dir, "segmentation")
#     os.makedirs(seg_dir, exist_ok=True)

# def main(args):
#     test_unetr(args, pred_dir=args.pred_dir)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base_dir", default="/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/")
#     parser.add_argument("--pred_dir", required=True)
#     args = parser.parse_args()

#     main(args)


data_path = '/scratch-grete/projects/nim00007/data/LiveCELL/'
patch_shape = (520, 704)

import torch
from torch.utils.data import Dataset

# ... (your existing code)

# Define the list of cell types
cell_types = [["A172"], ["BT474"], ["BV2"], ["Huh7"], ["MCF7"], ["SHSY5Y"], ["SkBr3"], ["SKOV3"]]

# Create a list to store datasets for each cell type
all_datasets = []

# Iterate over cell types and create datasets
for cell_type in cell_types:
    train_set = get_livecell_dataset(path=data_path,
        split="train",
        patch_shape=patch_shape,
        cell_types=cell_type,
        download=True,
        boundaries=True,
        #raw_transform = identity,
        n_samples=60

    )
    all_datasets.append(train_set)

# Create a ConcatDataset using all the individual datasets
concatenated_dataset = ConcatDataset(*all_datasets)
print(len(concatenated_dataset))

for batch in concatenated_dataset:
    # Extract the raw image from the batch (assuming raw is the first element in the batch)
    raw_image = batch[0]#[0]  # Assuming the first index corresponds to the raw image and we're extracting the first image in the batch

    # Check the value range of the raw image
    min_value = torch.min(raw_image).item()
    max_value = torch.max(raw_image).item()

    print(f"Min value: {min_value}, Max value: {max_value}")
    print(raw_image.shape)
    breakpoint()

    # Break the loop after processing the first batch (and first image in that batch)
    break


def identity(raw):
     return raw

train_set_A172 = get_livecell_dataset(
    path=data_path,
    split="train",
    patch_shape=patch_shape,
    cell_types=["A172"],                   
    download=True,
    boundaries=True,
    raw_transform = identity,
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
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
    num_workers=16,
    shuffle=True,
    n_samples=60

    )

train_dataset = ConcatDataset(train_set_A172, train_set_BT474, train_set_BV2, train_set_Huh7, train_set_MCF7, train_set_SHSY5Y, train_set_SkBr3, train_set_SKOV3)

# print(len(concatenated_dataset))
# #print(concatenated_dataset.dtype)

# for batch in concatenated_dataset:
#     # Extract the raw image from the batch (assuming raw is the first element in the batch)
#     raw_image = batch[0]#[0]  # Assuming the first index corresponds to the raw image and we're extracting the first image in the batch

#     # Check the value range of the raw image
#     min_value = torch.min(raw_image).item()
#     max_value = torch.max(raw_image).item()

#     print(f"Min value: {min_value}, Max value: {max_value}")

#     # Break the loop after processing the first batch (and first image in that batch)
#     break

