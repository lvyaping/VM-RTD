"""Explicit split manifests, one-to-one pairing and deterministic sample augmentation."""
import csv
import hashlib
import json
import io
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .checkpoints import sha256

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def make_manifest(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f'Manifest already exists: {output}')
    rows = []
    for split in ('train', 'val', 'test'):
        base = root / split
        imgs = [p for name in ('image', 'images') if (p := base/name).is_dir()]
        masks = [p for name in ('mask', 'masks') if (p := base/name).is_dir()]
        if len(imgs) != 1 or len(masks) != 1:
            raise ValueError(f'{base}: expected exactly one image(s) and mask(s) folder')
        image_files = sorted(p for p in imgs[0].iterdir() if p.suffix.lower() in EXTENSIONS)
        mask_files = sorted(p for p in masks[0].iterdir() if p.suffix.lower() in EXTENSIONS)
        mask_map = {}
        for p in mask_files:
            key = p.stem
            if key.startswith('mask_'):
                key = 'bus_' + key[len('mask_'):]
            key = key.replace('_mask_', '_')
            for suffix in ('_segmentation', '_mask'):
                if key.endswith(suffix):
                    key = key[:-len(suffix)]
                    break
            if key in mask_map:
                raise ValueError(f'Ambiguous masks for {key}; merge multiple lesion masks explicitly first')
            mask_map[key] = p
        used = set()
        for p in image_files:
            if p.stem in used or p.stem not in mask_map:
                raise ValueError(f'Duplicate image ID or missing mask: {p}')
            used.add(p.stem)
            rows.append(dict(split=split, sample_id=p.stem, image=p.relative_to(root).as_posix(),
                             mask=mask_map[p.stem].relative_to(root).as_posix(), patient_id=''))
        if set(mask_map) != used:
            raise ValueError(f'Unmatched masks in {split}: {sorted(set(mask_map)-used)[:10]}')
        if not image_files:
            raise ValueError(f'Empty split: {split}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['split','sample_id','image','mask','patient_id'])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'split','sample_id','image','mask','patient_id'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest must include split,sample_id,image,mask,patient_id')
        return list(reader)


def _inspect_pair(root, r):
    paths = [(root/r[k]).resolve() for k in ('image','mask')]
    try:
        for path in paths:
            path.relative_to(root)
    except ValueError:
        raise ValueError('Manifest paths must remain inside data_root') from None
    image_bytes, mask_bytes = (p.read_bytes() for p in paths)
    with Image.open(io.BytesIO(image_bytes)) as im, Image.open(io.BytesIO(mask_bytes)) as mask:
        if im.size != mask.size:
            raise ValueError(f'Image/mask size mismatch: {r["sample_id"]}')
        rgb = im.convert('RGB')
        decoded_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        pixels = np.asarray(mask.convert('L'))
        valid = ((pixels == 0) | (pixels == 1) | (pixels == 255)).all()
        if not valid or ((pixels == 1).any() and (pixels == 255).any()):
            raise ValueError(f'Nonbinary/ambiguous mask {paths[1]}')
    return decoded_hash, [hashlib.sha256(content).hexdigest() for content in (image_bytes,mask_bytes)]


def audit_manifest(root, manifest, strict=True, check_duplicates=True):
    root = Path(root).resolve()
    rows = read_manifest(manifest)
    counts = dict(train=0, val=0, test=0)
    seen_id, seen_content, seen_patient = {}, {}, {}
    digest = hashlib.sha256()
    warnings, errors, duplicates = [], [], []
    # Bounded IO concurrency; map preserves manifest order and fingerprint determinism.
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = []
        for index, item in enumerate(pool.map(lambda r: _inspect_pair(root,r), rows)):
            inspected.append(item)
            if (index+1) % 250 == 0:
                print(f'Audited {index+1}/{len(rows)} image-mask pairs', flush=True)
    for index, (r, (decoded_hash, file_hashes)) in enumerate(zip(rows, inspected)):
        split = r['split']
        if not r['sample_id'] or r['sample_id'] in ('.','..') or any(c in r['sample_id'] for c in ('/','\\')):
            raise ValueError('sample_id must be a plain nonempty filename stem')
        if split not in counts:
            raise ValueError(f'Unknown split {split}')
        counts[split] += 1
        if r['sample_id'] in seen_id:
            raise ValueError(f'Duplicate sample ID: {r["sample_id"]}')
        seen_id[r['sample_id']] = split
        if check_duplicates and decoded_hash in seen_content:
            other_split, other_id = seen_content[decoded_hash]
            duplicates.append({'first_id':other_id,'first_split':other_split,
                               'duplicate_id':r['sample_id'],'duplicate_split':split})
            if other_split != split:
                errors.append(f'Cross-split duplicate image: {other_id} ({other_split}) and {r["sample_id"]} ({split})')
            else:
                warnings.append(f'Duplicate image within {split}: {other_id}, {r["sample_id"]}')
        elif check_duplicates:
            seen_content[decoded_hash] = (split, r['sample_id'])
        patient = r['patient_id'].strip()
        if patient:
            if patient in seen_patient and seen_patient[patient] != split:
                raise ValueError(f'Patient appears in multiple splits: {patient}')
            seen_patient[patient] = split
        digest.update(json.dumps(r, sort_keys=True).encode())
        for file_hash in file_hashes:
            digest.update(file_hash.encode())
    if any(v == 0 for v in counts.values()):
        raise ValueError(f'All three independent splits required: {counts}')
    report = {'status':'failed' if errors else 'passed', 'errors':errors,'duplicates':duplicates,
            'duplicate_check':'enabled' if check_duplicates else 'disabled',
            'counts': counts, 'manifest_sha256': sha256(manifest),
            'data_fingerprint': digest.hexdigest(), 'warnings': warnings,
            'patient_check': 'complete' if all(r['patient_id'].strip() for r in rows) else 'incomplete: no patient identity claim',
            'checks': ['one_to_one', 'binary_masks', 'matching_dimensions'] + (['cross_split_decoded_image_duplicates'] if check_duplicates else [])}
    if strict and errors:
        raise ValueError('\n'.join(errors)+'\nRun audit to export the full diagnostic report; do not train this manifest.')
    return report


class SegmentationDataset(Dataset):
    def __init__(self, root, manifest, split, size=256, seed=42, augment=True, rotation=15.):
        self.root = Path(root)
        self.rows = [r for r in read_manifest(manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No samples in {split}')
        self.split, self.size, self.seed = split, size, seed
        self.augment, self.rotation, self.epoch = augment, rotation, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(self.root/r['image']) as im:
            image = im.convert('RGB')
        with Image.open(self.root/r['mask']) as im:
            mask_array = (np.asarray(im.convert('L')) > 0).astype(np.uint8)*255
            mask = Image.fromarray(mask_array)
        if self.split == 'train' and self.augment:
            token = f'{self.seed}:{self.epoch}:{r["sample_id"]}'.encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big'))
            if rng.random() < .5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if rng.random() < .5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            if rng.random() < .5:
                angle = rng.uniform(-self.rotation, self.rotation)
                image = image.rotate(angle, resample=getattr(Image, 'Resampling', Image).BILINEAR)
                mask = mask.rotate(angle, resample=getattr(Image, 'Resampling', Image).NEAREST)
        image = image.resize((self.size,self.size), getattr(Image, 'Resampling', Image).BILINEAR)
        if self.split == 'train':
            mask = mask.resize((self.size,self.size), getattr(Image, 'Resampling', Image).NEAREST)
        x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2,0,1).copy())/255.
        y = torch.from_numpy((np.array(mask)>0).astype(np.float32).copy()).unsqueeze(0)
        return x, y, r['sample_id']
