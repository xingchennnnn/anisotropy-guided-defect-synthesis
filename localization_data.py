from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms


def numeric_key(file_name: str):
    stem = os.path.splitext(file_name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def image_file(file_name: str) -> bool:
    return os.path.splitext(file_name)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_by_stem(file_names: list[str]) -> dict[str, str]:
    out = {}
    for name in file_names:
        stem = os.path.splitext(name)[0]
        out[stem] = name
        if stem.isdigit():
            out[str(int(stem))] = name
            out[stem.zfill(3)] = name
    return out


@dataclass
class PairIndex:
    image_path: str
    mask_path: str
    group_name: str = "defect"


def scan_image_mask_pairs(root: str) -> list[PairIndex]:
    pairs = []
    direct = collect_from_image_mask_dirs(root, "defect")
    if direct:
        return direct
    for group_name in sorted(os.listdir(root)):
        group_root = os.path.join(root, group_name)
        if not os.path.isdir(group_root):
            continue
        pairs.extend(collect_from_image_mask_dirs(group_root, group_name))
    return pairs


def collect_from_image_mask_dirs(root: str, group_name: str) -> list[PairIndex]:
    image_dir = os.path.join(root, "image")
    mask_dir = os.path.join(root, "mask")
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        return []
    image_files = sorted([f for f in os.listdir(image_dir) if image_file(f)], key=numeric_key)
    mask_files = sorted([f for f in os.listdir(mask_dir) if image_file(f)], key=numeric_key)
    image_map = index_by_stem(image_files)
    pairs = []
    for mask_name in mask_files:
        stem = os.path.splitext(mask_name)[0]
        image_name = image_map.get(stem) or image_map.get(stem.zfill(3))
        if image_name is None:
            continue
        pairs.append(PairIndex(os.path.join(image_dir, image_name), os.path.join(mask_dir, mask_name), group_name))
    return pairs


class LocalizationEvalDataset(Dataset):
    def __init__(self, root_dir: str, resize_shape: tuple[int, int] | None = (256, 256)):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.images = []
        for group_name in sorted(os.listdir(root_dir)):
            image_dir = os.path.join(root_dir, group_name)
            if not os.path.isdir(image_dir):
                continue
            files = sorted([f for f in os.listdir(image_dir) if image_file(f)], key=numeric_key)
            keep_from = len(files) // 3
            if group_name == "good":
                keep_from = len(files)
            self.images += [os.path.join(image_dir, f) for f in files[keep_from:]]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        group_dir, file_name = os.path.split(image_path)
        group_name = os.path.basename(group_dir)
        if group_name == "good":
            mask_path = None
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = self.resolve_mask_path(group_dir, file_name)
            has_anomaly = np.array([1], dtype=np.float32)
        image, mask = self.load_pair(image_path, mask_path)
        return {"image": image, "mask": mask, "has_anomaly": has_anomaly, "idx": idx}

    def resolve_mask_path(self, group_dir: str, file_name: str) -> str:
        mask_root = os.path.join(group_dir, "../../ground_truth", os.path.basename(group_dir))
        stem = os.path.splitext(file_name)[0]
        for name in (f"{stem}_mask.png", f"{stem}.png"):
            path = os.path.join(mask_root, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(file_name)

    def load_pair(self, image_path: str, mask_path: str | None):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else np.zeros(image.shape[:2])
        if self.resize_shape is not None:
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
        image = np.transpose((image / 255.0).astype(np.float32), (2, 0, 1))
        mask = (mask / 255.0).astype(np.float32)[None]
        return image, mask


class GeneratedMaskDataset(Dataset):
    def __init__(
        self,
        normal_dir: str,
        generated_root: str,
        length: int | None = None,
        image_size: int = 256,
        anomaly_probability: float = 0.5,
    ):
        self.normal_files = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if image_file(f)], key=numeric_key)
        self.pairs = scan_image_mask_pairs(generated_root)
        self.image_paths = [pair.image_path for pair in self.pairs]
        self.mask_paths = [pair.mask_path for pair in self.pairs]
        self.length = length or len(self.normal_files)
        self.anomaly_probability = anomaly_probability
        self.loader = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size], antialias=True)])

    def collect_pairs(self, root: str):
        pairs = scan_image_mask_pairs(root)
        return [pair.image_path for pair in pairs], [pair.mask_path for pair in pairs]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if torch.rand(()) > self.anomaly_probability or not self.mask_paths:
            image = self.loader(Image.open(self.normal_files[idx % len(self.normal_files)]).convert("RGB"))
            mask = torch.zeros(1, image.shape[-2], image.shape[-1])
            return {"image": image, "mask": mask, "has_anomaly": np.array([0], dtype=np.float32), "source": "normal"}

        pair_idx = idx % len(self.mask_paths)
        image = self.loader(Image.open(self.image_paths[pair_idx]).convert("RGB"))
        mask = (self.loader(Image.open(self.mask_paths[pair_idx]).convert("L")) > 0.5).float()
        has_anomaly = np.array([1 if mask.sum() > 0 else 0], dtype=np.float32)
        return {
            "image": image,
            "mask": mask,
            "has_anomaly": has_anomaly,
            "source": self.pairs[pair_idx].group_name if self.pairs else "defect",
        }

    def summary(self) -> dict:
        return {
            "normal_count": len(self.normal_files),
            "pair_count": len(self.pairs),
            "length": self.length,
            "groups": sorted(set(pair.group_name for pair in self.pairs)),
        }


def localization_collate(batch: list[dict]) -> dict:
    out = {}
    keys = set().union(*(sample.keys() for sample in batch))
    for key in keys:
        values = [sample.get(key) for sample in batch]
        try:
            out[key] = default_collate(values)
        except Exception:
            out[key] = values
    return out
