import numpy as np
import cv2

# Open a video capture object for the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Capture the first two frames to initialize the comparison
ret, frame1 = cap.read()  # Read the first frame
ret, frame2 = cap.read()  # Read the second frame

# The main loop that continuously processes each frame from the webcam feed
while cap.isOpened():
    
    # Calculate the absolute difference between the current frame and the previous frame
    # This is done to detect changes (movement) between consecutive frames.
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the difference image to grayscale to simplify processing
    # Grayscale conversion is done because motion detection is easier in a single channel.
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise and detail in the image
    # Blurring helps to remove noise and avoid false positives in motion detection.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to highlight significant changes in the image
    # Thresholding turns the image into black and white, where areas with movement are white.
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Perform dilation to make the white areas (motion) larger and more prominent
    # Dilation helps to close gaps in contours and make detected movement areas clearer.
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours in the dilated image (used to detect areas of movement)
    # Contours are used to detect continuous areas of white in the binary image (movement regions).
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the found contours
    for contour in contours:
        
        # Get the bounding rectangle for each contour
        # This will be used to draw a rectangle around the detected movement area.
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # If the contour area is too small, we skip it as it's not significant movement
        # Small movements (like tiny camera shifts) are ignored here.
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Draw a rectangle around the detected movement area
        # This rectangle highlights the region of motion in the frame.
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add a status text to indicate that movement is detected
        # "STATUS: Movement" is displayed on the frame, so the user knows motion was detected.
        cv2.putText(frame1, "STATUS:{}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    # Display the resulting frame with bounding boxes around detected movement
    # This allows the user to visually see the frame with detected movement areas marked.
    cv2.imshow('feed', frame1)
    
    # After processing the current frame, move to the next one
    # We update `frame1` to be the last processed frame, and `frame2` to the next captured frame.
    frame1 = frame2
    ret, frame2 = cap.read()  # Capture the next frame to process

    # Exit the loop if the user presses the "ESC" key (ASCII code 27)
    # This allows the user to stop the motion detection loop when desired.
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup: Release the video capture and close all windows
# Releasing the capture object and closing all OpenCV windows is necessary to free resources.
cv2.destroyAllWindows()
cap.release()
