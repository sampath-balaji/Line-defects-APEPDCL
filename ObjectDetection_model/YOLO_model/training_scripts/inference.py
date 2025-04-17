from ultralytics import YOLO
import os
from pathlib import Path
from PIL import Image


model_path = '/home/line_quality/line_quality/runs/detect/train225/weights/best.pt'
source_dir = '/home/line_quality/line_quality/custom_data'
output_dir = '/home/line_quality/line_quality/custom_data_predictions'
visualize = False 

model = YOLO(model_path)

os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in Path(source_dir).glob('*') if f.suffix.lower() in image_extensions]

# running inference on all images
for img_path in image_files:
    # run prediction
    results = model(img_path, conf=0.10)

    # save prediction image
    save_path = os.path.join(output_dir, img_path.name)
    results[0].save(filename=save_path)

    # display
    if visualize:
        img = Image.open(save_path)
        img.show()

print(f"Inference complete. Predictions saved to: {output_dir}")
