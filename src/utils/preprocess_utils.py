import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

def resize_with_padding(image, target_size=640):
    """
    Resizes an image while maintaining aspect ratio and adding padding.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    x_offset, y_offset = (target_size - new_w) // 2, (target_size - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded_image, scale, x_offset, y_offset

def adjust_yolo_labels(label_path, scale, x_offset, y_offset, orig_w, orig_h, target_size=640):
    """
    Reads YOLO labels, rescales them, and saves the adjusted labels.
    """
    new_bboxes = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = parts[0], *map(float, parts[1:])

            # Convert YOLO normalized format to absolute pixel values
            x_center_abs, y_center_abs = x_center * orig_w, y_center * orig_h
            width_abs, height_abs = width * orig_w, height * orig_h

            # Apply scaling and shifting
            x_center_new = (x_center_abs * scale + x_offset) / target_size
            y_center_new = (y_center_abs * scale + y_offset) / target_size
            width_new = (width_abs * scale) / target_size
            height_new = (height_abs * scale) / target_size

            new_bboxes.append(f"{class_id} {x_center_new} {y_center_new} {width_new} {height_new}\n")

    return new_bboxes

def process_dataset(original_img_dir, resized_img_dir, resized_labels_dir, yolo_labels_dir, target_size=640):
    """
    Resizes images, adjusts YOLO labels, and saves them.
    """
    for img_path in Path(original_img_dir).glob("*.jpg"):  # Adjust extension if needed
        img = cv2.imread(str(img_path))

        # Resize and pad image
        resized_img, scale, x_offset, y_offset = resize_with_padding(img, target_size)

        # Save resized image
        cv2.imwrite(str(Path(resized_img_dir) / img_path.name), resized_img)

        # Adjust YOLO labels
        label_path = Path(yolo_labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            new_bboxes = adjust_yolo_labels(label_path, scale, x_offset, y_offset, img.shape[1], img.shape[0], target_size)
            if new_bboxes:
                with open(Path(resized_labels_dir) / label_path.name, "w") as f:
                    f.writelines(new_bboxes)

        print(f"Resized {img_path.name}")


# augment functions

def yolo_to_coco(bbox, img_size):
    """
    Convert YOLO bbox [x_center, y_center, width, height] (normalized)
    to COCO bbox [x_min, y_min, width, height] (absolute), with clamping.
    """
    x_center, y_center, w, h = bbox
    # Convert to absolute coordinates
    x_min = (x_center - w/2) * img_size
    y_min = (y_center - h/2) * img_size
    x_max = (x_center + w/2) * img_size
    y_max = (y_center + h/2) * img_size
    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_size, x_max)
    y_max = min(img_size, y_max)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def coco_to_yolo(bbox, img_size):
    """
    Convert COCO bbox [x_min, y_min, width, height] (absolute)
    to YOLO bbox [x_center, y_center, width, height] (normalized).
    """
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    x_center = (x_min + x_max) / 2 / img_size
    y_center = (y_min + y_max) / 2 / img_size
    return [x_center, y_center, w / img_size, h / img_size]

def augment_images(resize_img_dir, resize_labels_dir,augmented_img_dir, augmented_labels_dir):
    """
    Applies augmentations to images and saves new images with updated labels.
    Bounding boxes are converted from YOLO (normalized) to COCO (absolute) for augmentation,
    then converted back to YOLO format.
    """
    img_size = 640
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Rotate(limit=10, p=0.5, border_mode=0),
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.8, 1.0),
                            ratio=(0.75, 1.33),
                            interpolation=1,
                            p=0.5),
        A.GaussianBlur(p=0.2),
    ], bbox_params=A.BboxParams(
        format="coco",  # working with absolute coordinates
        label_fields=["class_labels"],
        min_visibility=0.2
    ))

    image_files = list(Path(resize_img_dir).glob("*.jpg"))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        label_path = Path(resize_labels_dir) / (img_path.stem + ".txt")

        # Load YOLO labels and convert them to COCO format
        bboxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    yolo_bbox = list(map(float, parts[1:]))  # [x_center, y_center, w, h]
                    coco_bbox = yolo_to_coco(yolo_bbox, img_size)
                    bboxes.append(coco_bbox)
                    class_labels.append(class_id)

        if len(bboxes) == 0:
            print(f"Skipping {img_path.name} (no labels)")
            continue

        # Apply augmentations using COCO-format bboxes
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)

        # Save augmented image
        aug_img_path = Path(augmented_img_dir) / f"aug_{img_path.name}"
        cv2.imwrite(str(aug_img_path), augmented["image"])

        # Convert augmented bounding boxes from COCO to YOLO format and save them
        aug_label_path = Path(augmented_labels_dir) / f"aug_{img_path.stem}.txt"
        with open(aug_label_path, "w") as f:
            for bbox, class_id in zip(augmented["bboxes"], augmented["class_labels"]):
                yolo_bbox = coco_to_yolo(bbox, img_size)
                # Clamp values to [0, 1] for safety
                yolo_bbox = [max(0, min(1, v)) for v in yolo_bbox]
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        print(f"Augmented: {img_path.name} --> {aug_img_path.name}")

