import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO('loaded_bag.pt')#upload the pt file here

video_path = r"D:\vChanel\truck\truck_bag_1.avi"#path for the video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'truck_bag_1123.mp4'#name for the video to save  loading_area.mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define the zone coordinates
# zone = [(78,203), (1805,205), (1805,250), (81,257)]
# zone = [(1040,21), (1139,19), (1141,1026), (1038,1028)]#adjust the coordinates based on the video
# zone=[(593,203), (962,206), (963,244), (594,239)]
zone=[(1583,365), (1661,365), (1661,852), (1583,852)]



# Initialize a counter for objects entering the zone
object_count = 0
previous_count = 0

# Track frames where objects have been counted
recent_frames = []

# Create an output directory if it doesn't exist
output_folder = 'truck_bag_1(9/19/24)'
os.makedirs(output_folder, exist_ok=True)

# Function to check if a frame is recent
def is_frame_recent(frame_index, recent_frames):
    for i in recent_frames:
        if frame_index - i < 30:
            return True
    return False

frame_index = 0  # Initialize the frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes and centroids on detected persons with confidence >= 0.75
    for result in results:
        for box in result.boxes:
            # Filter by confidence
            if box.conf >= 0.50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)

                # Calculate the width and height of the bounding box
                width = x2 - x1
                height = y2 - y1

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)

                # Calculate and draw the centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 0), -1)

                # Display the confidence score
                confidence_text = f"{box.conf.item():.2f}"  # Convert tensor to float
                # cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Display the size of the bounding box
                size_text = f"W: {width} H: {height}"
                # cv2.putText(frame, size_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Check if the size of the bounding box is greater than 200x200
                if width > 100 and height > 100:
                    # Check if the centroid is inside the zone
                    if cv2.pointPolygonTest(np.array(zone, np.int32), (centroid_x, centroid_y), False) >= 0:
                        if not is_frame_recent(frame_index, recent_frames):
                            object_count += 1


                            recent_frames.append(frame_index)  # Mark this frame as counted
                            # Mark the object as counted by drawing a different color
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)

    # Draw the zone bounding box
    # cv2.polylines(frame, [np.array(zone, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the count
    cv2.putText(frame, f"Count: {object_count}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 255), 10)

    # Check if the object count has increased
    if object_count > previous_count:
        frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_filename, frame)
        previous_count = object_count  # Update the previous count

    # Write the frame to the output video
    out.write(frame)

    disp = cv2.resize(frame, (800, 800))
    # Display the frame
    cv2.imshow('Object Detection', disp)

    # Increment the frame index
    frame_index += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
