import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
from dataset_info import EVAL_IDS
import os
from pathlib import Path


# Pad an image to the target size using zero padding.
# The original image is placed in the top-left corner, and the
# original height and width are returned for later unpadding.
def pad(img, target_size):
    c, h, w = img.shape
    new_img = torch.zeros([3, target_size[0], target_size[1]])
    new_img[:, :h, :w] = img
    return new_img, h, w


class PairedDataset(torch.utils.data.Dataset):
    # from resize 576 x 1024 to padding 536 x 536
    def __init__(self, dataset_path, split, height=536, width=536, tokenizer=None, select_scenes=True, auto_size=False):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        
        # get dataset root
        self.dataset_root = str(Path(dataset_path).parent)

        # original dataset size
        org_size = len(self.data.keys())
        # key list
        ks = list(self.data.keys())

        # training-only filtering
        if split == "train" and select_scenes:
            for k in ks:
                # remove image pairs with severe degradation (LPIPS > 0.5)
                if self.data[k]["lpips"] > 0.5:
                    del self.data[k]
                    continue
                # remove images larger than 536×536 to reduce memory usage and simplify batching
                if int(self.data[k]["H"])>536 or int(self.data[k]["W"])>536:
                    del self.data[k]
                    continue

        # validation-only filtering
        if split != "train" and select_scenes:
            for k in ks:
                # remove images larger than 536×536 to simplify batching
                if int(self.data[k]["H"])>536 or int(self.data[k]["W"])>536:
                    del self.data[k]
                    continue
                # validating all image pairs is computationally expensive
                # we only validate image pairs specified in EVAL_IDS
                scene = self.data[k]["scene"]
                series = self.data[k]["series"]
                if f"{scene}_{series}" not in EVAL_IDS:
                    del self.data[k]
                    continue

        # new dataset size
        cur_size = len(self.data.keys())
        print("")
        print(f"{split}")
        print(f"Original dataset size: {org_size}")
        print(f"Current dataset size: {cur_size}")
        print("")


        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.auto_size = auto_size


    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = os.path.join(self.dataset_root, self.data[img_id]["image"])
        output_img = os.path.join(self.dataset_root, self.data[img_id]["target_image"])
        caption = self.data[img_id]["prompt"]

        scene_id, method, series = self.data[img_id]["scene"], self.data[img_id]["method"], self.data[img_id]["series"]
        dataset_id = self.data[img_id]["dataset"]

        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
        except Exception as e:
            raise RuntimeError(
                    f"Failed to load image pair:\n"
                    f"input: {input_img}\n"
                    f"target: {output_img}"
            ) from e

        # c, h, w
        img_t = F.to_tensor(input_img)
        c, h, w = img_t.shape
        # Automatically expand the target image size to the largest
        # multiple-of-8 resolution observed in the dataset.
        if self.auto_size:
            factor = 8
            h_new = ((h + factor - 1) // factor) * factor
            w_new = ((w + factor - 1) // factor) * factor
            self.image_size = (max(h_new, self.image_size[0]), max(w_new, self.image_size[1]))
        # pad image
        img_t, org_h, org_w = pad(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])


        # padding mask
        nopad_mask = torch.zeros((1, img_t.shape[1], img_t.shape[2]))
        nopad_mask[:, :org_h, :org_w] = 1
        
        # pad ground truth
        output_t = F.to_tensor(output_img)
        output_t, org_h, org_w = pad(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])
        
        # load image neighbors
        nn_path = self.data[img_id]["image"].rsplit("MODELS")[0]
        nn_path = os.path.join(self.dataset_root, nn_path, "NN", "nn_rank_by_name.json")
        with open(nn_path, "r") as f:
            cam_nn_all = json.load(f)

        # this depends on the dataset
        key_jpg = f"{series}.JPG"
        key_png = f"{series}.png"
        if key_jpg in cam_nn_all:
            cam_nn = cam_nn_all[key_jpg]
        elif key_png in cam_nn_all:
            cam_nn = cam_nn_all[key_png]
        else:
            raise KeyError(
                f"[cam_nn missing] Neither {key_jpg} nor {key_png} found in {nn_path}. "
                f"Available keys example: {list(cam_nn_all.keys())[:5]}"
        )
        cam_nn = [f"{scene_id}_{method}_{fn}".replace(".JPG", "").replace(".png", "") for fn in cam_nn if "extra" in fn]

        # find reference
        ref_name = "_source.png"
        path = os.path.join(self.dataset_root, self.data[img_id]["image"])
        parts = path.split("/")
        ref_img = None
        for idx in range(len(cam_nn)):
            nid = cam_nn[idx]
            candidate = "/".join(parts[:-1] + [nid + "_source.png"])
            if os.path.exists(candidate):
                ref_img = candidate
                break

        if ref_img is not None:
            # load reference image
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_img)
            ref_t, org_h, org_w = pad(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            # load reference ground truth
            output_ref_img = "/".join(parts[:-1] + [nid + "_target.png"])
            output_ref_img = Image.open(output_ref_img)
            output_ref_t = F.to_tensor(output_ref_img)
            output_ref_t, org_h, org_w = pad(output_ref_t, self.image_size)
            output_ref_t = F.normalize(output_ref_t, mean=[0.5], std=[0.5])
        else:
            # if no alternative reference image is available,
            # use the current image as its own reference.
            ref_t = img_t
            output_ref_t = output_t

        img_t = torch.stack([img_t, ref_t], dim=0)
        output_t = torch.stack([output_t, output_ref_t], dim=0)

        nopad_mask = nopad_mask.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "dataset_id": dataset_id,
            "scene_id": scene_id,
            "method": method,
            "series": series,
            "nopad_mask": nopad_mask.to(torch.bool),
            "org_h": org_h,
            "org_w": org_w,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
