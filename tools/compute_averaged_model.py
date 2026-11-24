import argparse
import os
import os.path as osp

import torch


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--start_epoch", type=int, required=True)
    parser.add_argument("--end_epoch", type=int, required=True)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    ckpt_dir = osp.join(args.exp_dir, "checkpoints")

    checkpoint = torch.load(f"{ckpt_dir}/epoch-{args.start_epoch}.pth", map_location="cpu")
    averaged_model = checkpoint["model"]

    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        checkpoint = torch.load(f"{ckpt_dir}/epoch-{epoch}.pth", map_location="cpu")
        model = checkpoint["model"]
        for key in averaged_model.keys():
            averaged_model[key] += model[key]

    num_epochs = args.end_epoch - args.start_epoch + 1
    for key in averaged_model.keys():
        averaged_model[key] /= num_epochs

    checkpoint = torch.load(f"{ckpt_dir}/epoch-{args.end_epoch}.pth", map_location="cpu")
    checkpoint["model"] = averaged_model
    torch.save(checkpoint, f"{ckpt_dir}/averaged_model.pth")


if __name__ == "__main__":
    main()
