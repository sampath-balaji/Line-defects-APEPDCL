from PIL import Image
from pathlib import Path
import os
import math

image_dir = '/home/line_quality/line_quality/custom_data_predictions'
output_dir = 'collages_output'
thumb_size = (320, 320)
grid_shape = (3, 3)

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted([p for p in Path(image_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
total_images = len(image_paths)

if total_images == 0:
    raise ValueError("No images found in the folder!")

# calculate number of collages needed
images_per_collage = grid_shape[0] * grid_shape[1]
num_collages = math.ceil(total_images / images_per_collage)

#  CREATE COLLAGES
for i in range(num_collages):
    collage_img = Image.new('RGB', (grid_shape[1]*thumb_size[0], grid_shape[0]*thumb_size[1]), color=(255, 255, 255))

    for j in range(images_per_collage):
        img_index = i * images_per_collage + j
        if img_index >= total_images:
            break  # means no more images left to add

        img = Image.open(image_paths[img_index]).convert('RGB')
        img = img.resize(thumb_size)

        row = j // grid_shape[1]
        col = j % grid_shape[1]
        x = col * thumb_size[0]
        y = row * thumb_size[1]

        collage_img.paste(img, (x, y))

    collage_path = os.path.join(output_dir, f'collage_{i+1}.jpg')
    collage_img.save(collage_path)
    print(f"✅ Saved collage {i+1}: {collage_path}")
