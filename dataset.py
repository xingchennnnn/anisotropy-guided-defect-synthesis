from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms


def _numeric_stem(stem: str) -> int | None:
    stem = stem[:-5] if stem.endswith("_mask") else stem
    return int(stem) if stem.isdigit() else None


class LineDefectDataset(Dataset):
    def __init__(
        self,
        root: str,
        source_name: str,
        image_size: tuple[int, int] = (256, 256),
        precomputed_direction_dir: str | None = None,
    ):
        self.root = Path(root) / source_name
        self.source_name = source_name
        self.image_size = image_size
        self.precomputed_direction_dir = Path(precomputed_direction_dir) if precomputed_direction_dir else None
        image_dir = self.root / "test" / "defect"
        mask_dir = self.root / "ground_truth" / "defect"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        image_names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        image_names = image_names[::3]

        mask_by_name = {}
        mask_by_index = {}
        if mask_dir.is_dir():
            for path in mask_dir.iterdir():
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                mask_by_name[path.name.lower()] = path
                index = _numeric_stem(path.stem.lower())
                if index is not None:
                    mask_by_index.setdefault(index, path)

        self.samples = []
        for name in image_names:
            image_path = image_dir / name
            stem = image_path.stem
            candidates = [name, f"{stem}.png", f"{stem}_mask{image_path.suffix}", f"{stem}_mask.png"]
            mask_path = next((mask_by_name[c.lower()] for c in candidates if c.lower() in mask_by_name), None)
            if mask_path is None and stem.isdigit():
                mask_path = mask_by_index.get(int(stem))
            self.samples.append((image_path, mask_path))

        self.image_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        if mask_path is None:
            mask = torch.zeros(1, *self.image_size)
        else:
            mask = (self.mask_transform(Image.open(mask_path).convert("L")) > 0.5).float()

        metadata = {
            "anomaly_filename": image_path.name,
            "anomaly_path": str(image_path),
            "mask_path": str(mask_path) if mask_path else None,
            "defect_type": "defect",
            "category": self.source_name,
            "idx": idx,
        }
        if self.precomputed_direction_dir:
            meta_path = self.precomputed_direction_dir / f"{idx:06d}.pt"
            if meta_path.exists():
                data = torch.load(meta_path, map_location="cpu", weights_only=False)
                metadata["precomputed_direction_map"] = data.get("direction_map", data.get("aniso_structure_map"))
        prompt = f"A sample from {self.source_name} with a defect"
        return image, mask, prompt, metadata


class MultiSourceDefectDataset(Dataset):
    def __init__(
        self,
        root: str,
        sources_config: list[dict],
        image_size: tuple[int, int] = (256, 256),
        precomputed_base_dir: str | None = None,
    ):
        self.datasets = []
        self.info = []
        self.cumulative_lengths = []
        total = 0
        for label_id, cfg in enumerate(sources_config):
            source_name = cfg.get("source_name", cfg.get("category"))
            defect_type = cfg.get("defect_type", "defect")
            if not source_name:
                raise ValueError("Each source config needs source_name")

            meta_dir = None
            if precomputed_base_dir:
                candidates = [
                    Path(precomputed_base_dir) / f"{source_name}_{defect_type}",
                    Path(precomputed_base_dir) / source_name,
                ]
                meta_dir = next((str(path) for path in candidates if path.exists()), None)

            dataset = LineDefectDataset(root, source_name, image_size, meta_dir)
            self.datasets.append(dataset)
            total += len(dataset)
            self.cumulative_lengths.append(total)
            self.info.append({"label_id": label_id, "source_name": source_name, "defect_type": defect_type, "length": len(dataset)})
        self.total_length = total

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int):
        dataset_idx = 0
        local_idx = idx
        for i, cumsum in enumerate(self.cumulative_lengths):
            if idx < cumsum:
                dataset_idx = i
                local_idx = idx - (self.cumulative_lengths[i - 1] if i > 0 else 0)
                break
        image, mask, prompt, metadata = self.datasets[dataset_idx][local_idx]
        metadata["global_idx"] = idx
        metadata["local_idx"] = local_idx
        metadata["class_label"] = self.info[dataset_idx]["label_id"]
        return image, mask, prompt, metadata

    def get_category_mapping(self) -> dict:
        category_to_label = {}
        label_to_category = {}
        for item in self.info:
            key = f"{item['source_name']}_{item['defect_type']}"
            category_to_label[key] = item["label_id"]
            label_to_category[item["label_id"]] = key
        return {"category_to_label": category_to_label, "label_to_category": label_to_category, "num_classes": len(self.info)}


def defect_collate(batch):
    images, masks, prompts, metadata = zip(*batch)
    output_meta = {}
    keys = set().union(*(m.keys() for m in metadata))
    for key in keys:
        if key == "precomputed_direction_map":
            if all(key in m and m[key] is not None for m in metadata):
                output_meta[key] = default_collate([m[key] for m in metadata])
            continue
        values = [m.get(key) for m in metadata]
        try:
            output_meta[key] = default_collate(values)
        except Exception:
            output_meta[key] = values
    return default_collate(images), default_collate(masks), list(prompts), output_meta
