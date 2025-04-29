import sys
import cv2
from inference import get_model
import supervision as sv

# Get video file name from command-line arguments
if len(sys.argv) != 2:
    print("Usage: python input.py <video_file>")
    sys.exit(1)

video_file = sys.argv[1]

# Open the video file
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error opening video file: {video_file}")
    sys.exit(1)

# Load a pre-trained yolov8n model
model = get_model(model_id="cv-900gl/1", api_key="ovg33Rnthku9c8XkBTZn")

# Create annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.infer(frame)[0]

    # Load results into Supervision Detections
    detections = sv.Detections.from_inference(results)

    # Optional: Create confidence labels
    labels = [
        f"{prediction.class_name} {prediction.confidence:.2f}"
        for prediction in results.predictions
    ]

    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Display the annotated frame
    cv2.imshow('Annotated Video', annotated_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
