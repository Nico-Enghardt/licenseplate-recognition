import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_boxes

# Load the model
model = DetectMultiBackend('runs/train/exp/weights/best.pt', device='cpu')  # Use 'cuda' for GPU

# Load an image
img = 'path/to/test_image.jpg'

# Run inference
results = model(img)

# Process results (e.g., display bounding boxes)
plot_boxes(results)
