import cv2

# Load the pre-trained model
model_path = 'path/to/your/model'
net = cv2.dnn.readNetFromTensorflow(model_path + '.pb', model_path + '.pbtxt')

# Load the image
image = cv2.imread('path/to/your/image.jpg')

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=True, crop=False)

# Pass the blob through the network
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter detections by confidence threshold
    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])

        # Get the bounding box coordinates
        height, width, _ = image.shape
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x_min, y_min, x_max, y_max = box.astype(int)

        # Draw the bounding box and label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f'Class ID: {class_id}, Confidence: {confidence:.2f}'
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the output
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
