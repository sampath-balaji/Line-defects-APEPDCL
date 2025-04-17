from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8m.pt')

# Device selection
device = 'cuda'  # or 'cpu'

# Best hyperparameters from tuning
best_hyp = {
    'lr0': 0.00772,
    'lrf': 0.01006,
    'momentum': 0.93729,
    'weight_decay': 0.00046,
    'warmup_epochs': 4.97639,
    'warmup_momentum': 0.69,
    'box': 6.18807,
    'cls': 0.48323,
    'dfl': 1.74065,
    'hsv_h': 0.01627,
    'hsv_s': 0.77614,
    'hsv_v': 0.49691,
    'degrees': 0.0,
    'translate': 0.07942,
    'scale': 0.50382,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.49787,
    'bgr': 0.0,
    'mosaic': 0.95964,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# Train the model
model.train(
    data=r"/home/line_quality/line_quality/yolotrainset/data.yaml",  # dataset path
    epochs=100,
    batch=16,
    imgsz=640,
    patience=50,
    device=device,
    half=True,
    workers=0,
    **best_hyp
)

#Validate after training
metrics = model.val()
print(metrics)