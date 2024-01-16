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


def visualize_sample(inputs, segmentation, boundaries, filename):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(inputs, cmap="gray")
    axs[0].set_title("Input")

    axs[1].imshow(segmentation, cmap="viridis")
    axs[1].set_title("Segmentation")

    axs[2].imshow(boundaries, cmap="gray")
    axs[2].set_title("Boundaries")

    plt.savefig(os.path.join("/home/nimmahen/code/plots", f"visualization_standrd_{filename}.png"))
    plt.close()


def test_unetr(args, model_weights: str, pred_dir: str):
    cell_types = ["[MCF7]"]
    for i in cell_types:
        test_images = os.path.join("/scratch-grete/projects/nim00007/data/LiveCELL/images/livecell_test_images/", i + "*")

        model = UNETR(
            backbone=args.backbone, encoder=args.encoder, out_channels=2, use_sam_stats=args.use_sam_stats,
            final_activation="Sigmoid",
        )

        from micro_sam.util import _CustomUnpickler
        import pickle

        custom_pickle = pickle
        custom_pickle.Unpickler = _CustomUnpickler

        state = torch.load(model_weights, map_location="cpu", pickle_module=custom_pickle)
        model_state = state["model_state"]
        model.load_state_dict(model_state)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model.to(device)
        model.eval()

        with torch.no_grad():

            for img_path in glob.glob(test_images):
                filename = os.path.split(img_path)[-1]

                inputs = imageio.imread(img_path)
                inputs = inputs.astype(np.float32)


                
                # ...
                #inputs = torch_em.transform.raw.standardize(inputs)
                predictions = predict_with_halo(inputs, model, gpu_ids=[device], block_shape=(384, 384), halo=(68, 68), preprocess=None)

                                                
                # ...

                segmentation = predictions[0, :, :]
                boundaries = predictions[1, :, :]

                # Visualize and save the segmentation and boundaries
                visualize_sample(inputs, segmentation, boundaries, filename)
                breakpoint()



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

# --encoder "vit_b" --backbone "sam" --model_weights /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/livecell_MCF7_10_vit_b_Btch2/checkpoints/unetr-MCF7/best.pt --pred_dir livecell_MCF7_10_vit_b_Btch2
