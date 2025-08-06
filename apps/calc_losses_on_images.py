"""Evaluate reconstruction metrics on a folder of generated images.

This script computes LPIPS, L2 or mIOU scores by comparing generated results
to ground-truth images.  It has been rewritten with clear structure and
documentation to make it easier for new users to run.
"""

from argparse import ArgumentParser
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from finetune_hybrid_encoder import face_parsing, initFaceParsing
from inversion.criteria.lpips.lpips import LPIPS
from inversion.datasets.gt_res_dataset import GTResDataset


class MIOU(torch.nn.Module):
    """Mean intersection-over-union for facial semantic masks."""

    def __init__(self) -> None:
        super().__init__()
        self.parser, _ = initFaceParsing(path="pretrained_models", device=None)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = face_parsing(source, self.parser)
        target = face_parsing(target, self.parser)
        return torch.mean(
            torch.div(
                torch.sum(source * target, dim=[2, 3]).float(),
                torch.sum((source + target) > 0, dim=[2, 3]).float() + 1e-6,
            ),
            dim=1,
        )


def parse_args():
    parser = ArgumentParser(description="Compute reconstruction metrics.")
    parser.add_argument("--mode", choices=["lpips", "l2", "miou"], default="lpips")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--gt_path", type=str, default="gt_images")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def run(args):
    for step in sorted(os.listdir(args.output_path)):
        yaw = float(step.split("_")[-1])
        if yaw < 1.9:
            continue
        output_path = os.path.join(args.output_path, step, "can")
        gt_path = os.path.join(args.gt_path, step, "gt")
        run_on_step_output(output_path, gt_path, args)


def run_on_step_output(output_path: str, gt_path: str, args) -> None:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    print("Loading dataset")
    dataset = GTResDataset(root_path=output_path, gt_dir=gt_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True,
    )

    if args.mode == "lpips":
        loss_func = LPIPS(net_type="alex")
    elif args.mode == "l2":
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = MIOU()
    loss_func.cuda()

    scores, scores_dict, idx = [], {}, 0
    for result_batch, gt_batch in tqdm(dataloader):
        for i in range(args.batch_size):
            loss = float(
                loss_func(result_batch[i : i + 1].cuda(), gt_batch[i : i + 1].cuda())
            )
            scores.append(loss)
            im_path = dataset.pairs[idx][0]
            scores_dict[os.path.basename(im_path)] = loss
            idx += 1

    mean, std = np.mean(scores), np.std(scores)
    result_str = f"Average loss is {mean:.2f}+/-{std:.2f}"
    print("Finished with", output_path)
    print(result_str)

    out_path = os.path.join(os.path.dirname(output_path), "inference_metrics")
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, f"stat_{args.mode}_can.txt"), "w") as f:
        f.write(result_str)
    with open(os.path.join(out_path, f"scores_{args.mode}_can.json"), "w") as f:
        json.dump(scores_dict, f)


if __name__ == "__main__":
    sys.path.extend([".", ".."])
    run(parse_args())

