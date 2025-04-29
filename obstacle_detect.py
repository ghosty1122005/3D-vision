import sys
import cv2
import numpy as np
from inference import get_model
import supervision as sv

# ------------------ Helper Functions ------------------

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def match_object(bbox, previous_objects, max_dist=50):
    c1 = centroid(bbox)
    best_id = None
    best_dist = float('inf')
    for obj_id, data in previous_objects.items():
        c2 = centroid(data['bbox'])
        dist = np.linalg.norm(np.array(c1) - np.array(c2))
        if dist < max_dist and dist < best_dist:
            best_id = obj_id
            best_dist = dist
    return best_id

# ------------------ Argument Check ------------------

if len(sys.argv) != 2:
    print("Usage: python input.py <video_file>")
    sys.exit(1)

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error opening video file: {video_file}")
    sys.exit(1)

# ------------------ Load Model ------------------

model = get_model(model_id="cv-900gl/1", api_key="ovg33Rnthku9c8XkBTZn")

# ------------------ Annotators ------------------

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ------------------ Tracking Setup ------------------

previous_objects = {}
object_id_counter = 0

GROWTH_THRESHOLD = 0.1      # 10% increase in area indicates approach
AREA_THRESHOLD = 1000          # Minimum area to consider an object
MAX_AREA_THRESHOLD = 500000 # Large bounding box threshold
frame_skip = 1              # Skip every 3 frames
frame_count = 0

# ------------------ Main Loop ------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    current_objects = {}

    for pred in results.predictions:
        # Convert bbox from center format to (x1, y1, x2, y2)
        x1 = int(pred.x - pred.width / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width / 2)
        y2 = int(pred.y + pred.height / 2)
        bbox = [x1, y1, x2, y2]

        area = (x2 - x1) * (y2 - y1)
        label = pred.class_name

        # Match with previous objects
        matched_id = match_object(bbox, previous_objects)
        if matched_id is None:
            matched_id = object_id_counter
            object_id_counter += 1

        is_obstacle = False
        if matched_id in previous_objects:
            prev_area = previous_objects[matched_id]['area']
            growth = (area - prev_area) / (prev_area + 1e-5)
            print(f"ID {matched_id} | Area: {area} | Growth: {growth:.2f}")

            if (growth > GROWTH_THRESHOLD and area > AREA_THRESHOLD) or area > MAX_AREA_THRESHOLD:
                is_obstacle = True
                label += " (Obstacle!)"
                reason = "growth" if growth > GROWTH_THRESHOLD else "size"
                print(f"ID {matched_id} marked as obstacle due to {reason} | Area: {area}")

        current_objects[matched_id] = {'bbox': bbox, 'area': area}

        # Draw bounding box and label
        color = (0, 0, 255) if is_obstacle else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update state
    previous_objects = current_objects.copy()

    # Display output
    cv2.imshow('Obstacle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ Cleanup ------------------

cap.release()
cv2.destroyAllWindows()
