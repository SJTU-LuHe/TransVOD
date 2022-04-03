# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, interval1, interval2, num_ref_frames= 3,
        is_train = True,  filter_key_img=True,  cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # idx若为675834，则img_id为675835(img_id=idx+1)
        imgs = []
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']
        img = self.get_image(path)
        target = {'image_id': img_id, 'annotations': target}
        img, target = self.prepare(img, target)
        imgs.append(img)
        if video_id == -1: # imgnet_det
            for i in range(self.num_ref_frames):
                imgs.append(img)
        else: # imgnet_vid 
            img_ids = self.cocovid.get_img_ids_from_vid(video_id) 
            ref_img_ids = []
            if self.is_train: # Train
                interval = self.num_ref_frames + 2 # *20
                left = max(img_ids[0], img_id - interval)
                right = min(img_ids[-1], img_id + interval)
                sample_range = list(range(left, right+1))
                if self.num_ref_frames >= 10:
                    sample_range=img_ids
                if self.filter_key_img and img_id in sample_range:
                    sample_range.remove(img_id) 
                while len(sample_range) < self.num_ref_frames:
                    print("sample_range", sample_range)
                    sample_range.extend(sample_range)
                ref_img_ids = random.sample(sample_range, self.num_ref_frames)

            else:  # Eval
                ref_img_ids = []
                Len = len(img_ids)
                interval  = max(int(Len // 16), 1)
                if self.num_ref_frames < 8:
                    left_indexs = int((img_id - img_ids[0]) // interval)
                    right_indexs = int((img_ids[-1] - img_id) // interval)
                    if left_indexs < self.num_ref_frames:
                        for i in range(self.num_ref_frames):
                            ref_img_ids.append(min(img_id + (i+1)*interval, img_ids[-1]))
                    else:
                        for i in range(self.num_ref_frames):
                            ref_img_ids.append(max(img_id - (i+1)* interval, img_ids[0]))
                sample_range = []
                if self.num_ref_frames >= 8:
                    left_indexs = int((img_ids[0] - img_id) // interval)
                    right_indexs = int((img_ids[-1] - img_id) // interval)
                    for i in range(left_indexs, right_indexs):
                        if i < 0:
                            index = max(img_id + i*interval, img_ids[0])
                            sample_range.append(index)
                        elif i > 0:
                            index = min(img_id + i * interval, img_ids[-1])
                            sample_range.append(index)
                    if self.filter_key_img and img_id in sample_range:
                        sample_range.remove(img_id)
                    while len(sample_range) < self.num_ref_frames:
                        print("sample_range", sample_range)
                        sample_range.extend(sample_range)
                    ref_img_ids = sample_range[:self.num_ref_frames]

            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_img = self.get_image(ref_img_path)
                imgs.append(ref_img)

        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target) 
        
        return  torch.cat(imgs, dim=0),  target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        "train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), is_train =(not args.eval), interval1=args.interval1,
                                interval2=args.interval2, num_ref_frames = args.num_ref_frames, return_masks=args.masks, cache_mode=args.cache_mode, 
                                local_rank=get_local_rank(), local_size=get_local_size())
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    
