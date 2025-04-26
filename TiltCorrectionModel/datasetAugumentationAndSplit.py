import glob
import random
import os
import math
from PIL import Image

# Set the random seed for reproducibility
random.seed(0)

# Define input and output directories
input_dir = '8apr_dataset'  # Folder containing all original images
output_dir = '8apr_dataset_split'

# Define the splits and the rotation angles (in degrees)
splits = ['train', 'val', 'test']
rotation_angles = [0, 2.5, -2.5, 5, -5]

# Create output directories for each split and each rotation angle
for split in splits:
    for angle in rotation_angles:
        target_dir = os.path.join(output_dir, split, str(angle))
        os.makedirs(target_dir, exist_ok=True)

# Get all jpg images from the input directory
all_images = glob.glob(os.path.join(input_dir, '*.jpg'))
print(f"Found {len(all_images)} images in '{input_dir}'.")

if not all_images:
    print("No images found in the input directory. Check the folder name and path.")
    exit()

# Shuffle and split the images: 75% train, 15% val, 10% test
random.shuffle(all_images)
total_images = len(all_images)
num_train = round(0.75 * total_images)
num_test = round(0.10 * total_images)
num_val = total_images - num_train - num_test
print(f"Total images: {total_images}. Splitting into {num_train} train, {num_val} val, {num_test} test.")

train_images = all_images[:num_train]
test_images = all_images[num_train:num_train + num_test]
val_images = all_images[num_train + num_test:]

# --- Rotation Utilities ---

def largest_rotated_rect(w, h, angle_rad):
    """
    Calculates the width and height of the largest possible axis-aligned rectangle
    within the rotated image.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))

    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        if width_is_longer:
            wr = x / sin_a
            hr = x / cos_a
        else:
            wr = x / cos_a
            hr = x / sin_a
    else:
        cos_2a = cos_a ** 2 - sin_a ** 2
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    return int(wr), int(hr)

def rotate_and_crop(img, angle):
    orig_w, orig_h = img.size

    # Rotate with expand=True
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Compute the largest safe inner rectangle
    angle_rad = math.radians(angle)
    new_w, new_h = largest_rotated_rect(orig_w, orig_h, angle_rad)

    # Center crop
    cx, cy = rotated.size[0] // 2, rotated.size[1] // 2
    left = cx - new_w // 2
    top = cy - new_h // 2
    right = cx + new_w // 2
    bottom = cy + new_h // 2

    cropped = rotated.crop((left, top, right, bottom))

    # Resize back to original dimensions
    final = cropped.resize((orig_w, orig_h), Image.LANCZOS)
    return final

# --- Main Image Processing Function ---

def process_images(image_list, split_name):
    for image_file in image_list:
        try:
            img = Image.open(image_file).convert('RGB')  # Ensure consistent mode
        except Exception as e:
            print(f"Error opening {image_file}: {e}")
            continue

        base_name, ext = os.path.splitext(os.path.basename(image_file))

        for angle in rotation_angles:
            final_img = rotate_and_crop(img, angle)
            new_filename = f"{base_name}_rot{angle}{ext}"
            target_path = os.path.join(output_dir, split_name, str(angle), new_filename)
            final_img.save(target_path)

        print(f"Processed and augmented {os.path.basename(image_file)} for split '{split_name}'.")

# --- Run Processing ---

print("\nProcessing training images...")
process_images(train_images, 'train')

print("\nProcessing validation images...")
process_images(val_images, 'val')

print("\nProcessing test images...")
process_images(test_images, 'test')

print("✅ Dataset preparation complete — tilt-ready and borderless!")
