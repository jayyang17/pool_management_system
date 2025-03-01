import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random
import shutil
from src.utils.utils import  write_yaml

# Define a constant for the target size
TARGET_SIZE = 640

def resize_with_padding(image, target_size=TARGET_SIZE):
    """
    Resizes an image while maintaining aspect ratio and adding padding.
    Returns the padded image, scale factor, x and y offsets,
    new dimensions, and the original width and height.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image while keeping the aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank image with a gray background
    padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Center the resized image in the padded image
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded_image, scale, x_offset, y_offset, new_w, new_h, w, h

def adjust_yolo_labels(label_path, scale, x_offset, y_offset, orig_w, orig_h, target_size=TARGET_SIZE):
    """
    Reads YOLO labels from the given path, rescales them for the new image size,
    and returns the adjusted labels as a list of strings.
    """
    new_bboxes = []
    if not label_path.exists():
        return []

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip lines that don't match expected format

            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert normalized YOLO coordinates to absolute pixel values
            x_center_abs = x_center * orig_w
            y_center_abs = y_center * orig_h
            width_abs = width * orig_w
            height_abs = height * orig_h

            # Scale and shift the coordinates for the resized image
            x_center_new = (x_center_abs * scale + x_offset) / target_size
            y_center_new = (y_center_abs * scale + y_offset) / target_size
            width_new = (width_abs * scale) / target_size
            height_new = (height_abs * scale) / target_size

            # Clamp values to [0, 1]
            x_center_new = max(0, min(1, x_center_new))
            y_center_new = max(0, min(1, y_center_new))
            width_new = max(0, min(1, width_new))
            height_new = max(0, min(1, height_new))

            new_bboxes.append(f"{class_id} {x_center_new} {y_center_new} {width_new} {height_new}\n")

    return new_bboxes

def process_dataset(original_img_dir, resized_img_dir, resized_labels_dir, yolo_labels_dir):
    """
    Processes a dataset by resizing images with padding and adjusting corresponding YOLO labels.
    Ensures that destination directories exist.
    """
    # Create destination directories if they don't exist
    os.makedirs(resized_img_dir, exist_ok=True)
    os.makedirs(resized_labels_dir, exist_ok=True)

    # Allow common image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(original_img_dir).glob(ext))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path.name}")
            continue

        # Resize and pad image
        (resized_img, scale, x_offset, y_offset, new_w, new_h, 
         orig_w, orig_h) = resize_with_padding(img, TARGET_SIZE)

        # Save resized image
        resized_img_path = Path(resized_img_dir) / img_path.name
        cv2.imwrite(str(resized_img_path), resized_img)

        # Adjust YOLO labels if the label file exists
        label_path = Path(yolo_labels_dir) / (img_path.stem + ".txt")
        new_bboxes = adjust_yolo_labels(label_path, scale, x_offset, y_offset, orig_w, orig_h, TARGET_SIZE)

        # Save updated labels if any bounding boxes were adjusted
        if new_bboxes:
            resized_label_path = Path(resized_labels_dir) / (img_path.stem + ".txt")
            with open(resized_label_path, "w") as f:
                f.writelines(new_bboxes)

        print(f"Processed {img_path.name}")

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
    os.makedirs(augmented_img_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)

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


def balance_dataset(
        resized_img_dir,
        resized_labels_dir,
        augmented_img_dir,
        augmented_labels_dir,
        training_img_dir,
        training_labels_dir,
        ratio_original=0.3, 
        total_samples=None):
    """
    Creates a balanced training dataset by copying a weighted mix of original and augmented images and labels.
    If total_samples is not specified, use all available images
    """

    # Create training directories if they don't exist
    os.makedirs(training_img_dir, exist_ok=True)
    os.makedirs(training_labels_dir, exist_ok=True)

    # Get lists of images
    orig_images = list(Path(resized_img_dir).glob("*.jpg"))
    aug_images  = list(Path(augmented_img_dir).glob("*.jpg"))

    # If total_samples is not specified, use all available images
    if total_samples is None:
        total_samples = len(orig_images) + len(aug_images)

    # Compute the number of images for each set
    num_orig = int(total_samples * ratio_original)
    num_aug  = total_samples - num_orig

    # Sample images from each list; if not enough images exist, take them all
    chosen_orig = orig_images if len(orig_images) <= num_orig else random.sample(orig_images, num_orig)
    chosen_aug  = aug_images  if len(aug_images)  <= num_aug  else random.sample(aug_images, num_aug)

    # Copy original images and their labels
    for img_path in chosen_orig:
        shutil.copy(str(img_path), training_img_dir)
        label_file = Path(resized_labels_dir) / (img_path.stem + ".txt")
        if label_file.exists():
            shutil.copy(str(label_file), training_labels_dir)

    # Copy augmented images and their labels
    for img_path in chosen_aug:
        shutil.copy(str(img_path), training_img_dir)
        label_file = Path(augmented_labels_dir) / (img_path.stem + ".txt")
        if label_file.exists():
            shutil.copy(str(label_file), training_labels_dir)

    print(f"Balanced dataset created with {len(chosen_orig)} original and {len(chosen_aug)} augmented images.")

def split_train_val(input_image_path, input_label_path, train_percent=0.8, output_dir=os.getcwd()):
    """
    Splits image and annotation files into train and validation sets.
    """
    if not (0.01 <= train_percent <= 0.99):
        raise ValueError(f"Invalid train_percent: {train_percent}. Must be between 0.01 and 0.99")

    # Check if input directories exist
    if not os.path.isdir(input_image_path):
        raise FileNotFoundError(f"Image directory not found: {input_image_path}")
    if not os.path.isdir(input_label_path):
        raise FileNotFoundError(f"Label directory not found: {input_label_path}")

    # Define paths to train/val images and labels
    
    train_img_path = os.path.join(output_dir,'data/train/images')
    train_txt_path = os.path.join(output_dir,'data/train/labels')
    val_img_path = os.path.join(output_dir,'data/validation/images')
    val_txt_path = os.path.join(output_dir,'data/validation/labels')

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of image and annotation files
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    img_file_list = [f for f in Path(input_image_path).rglob("*") if f.suffix.lower() in img_extensions]
    txt_file_list = [f for f in Path(input_label_path).rglob("*") if f.suffix.lower() == ".txt"]

    print(f"Found {len(img_file_list)} images and {len(txt_file_list)} labels.")

    # Determine number of files for train and validation
    train_num = int(len(img_file_list) * train_percent)
    val_num = len(img_file_list) - train_num
    print(f"Splitting data: {train_num} for training, {val_num} for validation.")

    # Shuffle and split files
    random.seed(42) 
    random.shuffle(img_file_list)
    train_files = img_file_list[:train_num]
    val_files = img_file_list[train_num:]

    # Copy files to train/val directories
    for img_path, target_dir in zip([train_files, val_files], [train_img_path, val_img_path]):
        for img in img_path:
            img_fn = img.name
            base_fn = img.stem
            txt_fn = base_fn + ".txt"
            txt_path = os.path.join(input_label_path, txt_fn)

            if not os.path.exists(txt_path):
                print(f"Label file not found for image {img_fn}. Skipping.")
                continue

            label_target_dir = train_txt_path if target_dir == train_img_path else val_txt_path
            try:
                shutil.copy(img, os.path.join(target_dir, img_fn))  # Copy image
                shutil.copy(txt_path, os.path.join(label_target_dir, txt_fn))  # Copy label
            except Exception as e:
                print(f"Error copying file {img_fn}: {e}")

    print("Data split complete! Check the 'train/' and 'validation/' folders.")

def create_yolo_data_yaml(
        data_path,
        classes_txt_path,  
        output_yaml_path,
        train_path="train/images",
        val_path="validation/images"): 
    
    """Creates a YOLO data.yaml file."""
    try:
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        
        data = {
            'path': data_path, 
            'train': train_path,  
            'val': val_path, 
            'nc': len(classes),
            'names': classes
        }

        write_yaml(data, output_yaml_path)
        print(f"YAML file created at {output_yaml_path}")

    except FileNotFoundError:
        print(f"File not found: {classes_txt_path}")
    except Exception as e:
        print(f"Error: {e}")
