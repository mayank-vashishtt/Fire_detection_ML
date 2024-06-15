import cv2
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Initialize OpenCV capture object
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Load image processor and model
processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")

# Function to process each frame from the webcam
def process_frame(frame):
    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(rgb_frame)
    
    # Preprocess the image using the processor
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction using the model
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get predicted label
    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_label]
    
    return predicted_class, rgb_frame

# Main loop to capture frames from webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Process the frame and get prediction
    predicted_class, rgb_frame = process_frame(frame)
    
    # Display the result on the frame
    cv2.putText(frame, predicted_class, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw bounding box around fire (if detected)
    if predicted_class == "fire":
        # Perform additional processing or use the model output to draw bounding box
        # For simplicity, let's draw a fixed bounding box in the center of the frame
        height, width, _ = frame.shape
        x1, y1 = int(width * 0.25), int(height * 0.25)
        x2, y2 = int(width * 0.75), int(height * 0.75)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw rectangle
        
    # Display the frame
    cv2.imshow('Fire Detection', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
