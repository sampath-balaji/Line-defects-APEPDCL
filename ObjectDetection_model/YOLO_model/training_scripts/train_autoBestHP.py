import os
import torch
from ultralytics import YOLO
import yaml

# Fix OpenMP issue (avoids Intel MKL conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure CUDA (GPU) is used, else fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#Load YOLO Model
model = YOLO('yolov8m.pt')

#Auto-Find Best Params
print("🛠️ Running Auto Hyperparameter Tuning...")
best_hyp = model.tune(
    data = r"/home/line_quality/line_quality/yolotrainset/data.yaml",  # Dataset path
    epochs=30,       # Run for 30 tuning epochs
    iterations=50,   # Try 50 different hyperparameter sets
    optimizer="AdamW",  # Use AdamW optimizer
    verbose=True     # Show detailed tuning logs
)

# Load best hyperparameters from saved YAML
with open("runs/detect/tune6/best_hyperparameters.yaml", "r") as f:
    best_hyp = yaml.safe_load(f)

# Train the model using best hyperparameters found
print("Training YOLOv8 with Best Found Hyperparameters...")
model.train(
    data = r"/home/line_quality/line_quality/yolotrainset/data.yaml",  # Dataset path
    epochs=80,
    batch=16,     
    imgsz=640,
    patience=50,   
    device=device, 
    half=True,     
    save=True,
    workers=0,     
    **best_hyp      
)

#Run Validation
print("📊 Running validation on trained model...")
metrics = model.val()

#Export/Save Model
#model.export(format="onnx")
model.save("best_model.pt")
