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

def resize_images(original_img_dir, resized_img_dir, resized_labels_dir, yolo_labels_dir, target_size=640):
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