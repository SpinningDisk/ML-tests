import tensorflow as tf
import tensorflow_hub as hub

# Load EfficientDet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Load and preprocess image
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img / 255
    img = tf.image.resize(img, [512, 512])
    img = tf.image.convert_image_dtype(img, tf.uint8)
    return img

image_path = 'IMG_0362.png'
image = load_img(image_path)
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# Run model inference
detections = model(image)

# Process and display detections
import numpy as np
import matplotlib.pyplot as plt

#COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
#                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
#                "scissors", "teddy bear", "hair drier", "toothbrush"]
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'dog', 'cat', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner', 'blanket', 'branch',
    'bridge', 'bus stop', 'butterfly', 'candle', 'chopsticks', 'computer keyboard', 'light switch', 'mirror',
    'office chair', 'paintbrush', 'picture frame', 'plate', 'playing card', 'poster', 'snowman', 'squirrel',
    'stool', 'street sign', 'table', 'tree'
]

boxes = detections['detection_boxes'].numpy()
class_names = detections['detection_classes'].numpy()
scores = detections['detection_scores'].numpy()

print(max(class_names[0]))

def display_detections(image, boxes, class_names, scores, max_boxes=10, min_score=0):
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0])
    ax = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
    
    for i in range(min(max_boxes, boxes.shape[1])):
        if scores[0, i] >= min_score:
            box = boxes[0, i]
            class_name = class_names[0, i]
            score = scores[0, i]
            
            y1, x1, y2, x2 = box
            ax.add_patch(plt.Rectangle((x1 * image.shape[2], y1 * image.shape[1]),
                                       (x2 - x1) * image.shape[2],
                                       (y2 - y1) * image.shape[1],
                                       fill=False, edgecolor=colors[int(class_name) % len(colors)], linewidth=2))
            text = f"{COCO_CLASSES[int(class_name)-1]}: {score:.2f}"
            ax.text(x1 * image.shape[2], y1 * image.shape[1] - 2, text, bbox=dict(facecolor='yellow', alpha=0.5),
                    fontsize=12, color='black')
    
    plt.show()
    plt.savefig("boxes.png")

# Assuming COCO class names for illustration purposes




display_detections(image, boxes, class_names, scores)
