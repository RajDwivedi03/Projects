import numpy as np           # Importing NumPy, a library for numerical operations (not directly used here but essential for many OpenCV operations)
import cv2                   # Importing OpenCV, which is the core library used for image and video processing tasks

# Load pre-trained Haar Cascade Classifiers for face and eye detection.
# Haar Cascade Classifiers are machine learning models trained using positive and negative image samples.
# They are very efficient and fast for object detection, particularly in real-time applications like face detection.
# These classifiers work by detecting patterns or features in images that match trained models.
# Alternative: Instead of Haar cascades, you can use advanced deep learning models like YOLO (You Only Look Once) or SSD (Single Shot Detector).
# These models offer better accuracy, especially for detecting faces in varying conditions like different angles, lighting, etc.
face_cascade = cv2.CascadeClassifier('C:/Users/rajdw/Favorites/Downloads/haarcascade_frontalface_default.xml')  # Face detection model
eye_cascade = cv2.CascadeClassifier('C:/Users/rajdw/Favorites/Downloads/haarcascade_eye_tree_eyeglasses.xml')   # Eye detection model

# Start capturing video from the webcam (by default, the first webcam is used with index 0)
# OpenCV provides an easy interface for accessing the webcam using VideoCapture.
# If you want to use a different camera, change the index (e.g., 'cv2.VideoCapture(1)' for another connected camera).
cap = cv2.VideoCapture(0)  # Initialize webcam capture object, '0' refers to the default camera

# Check if the webcam is successfully opened
# 'cap.isOpened()' returns True if the webcam is ready for capturing frames; otherwise, it returns False.
# If not opened, print an error message and exit the program. This is important to prevent the program from running without access to the webcam.
if not cap.isOpened():
    print("Error: Could not access the webcam.")   # Inform the user that the webcam is not accessible
    exit()    # Exit the program if webcam access is unsuccessful

# Main loop that runs while the webcam is active
# This loop continuously grabs frames from the webcam, processes the image, and displays the result.
while cap.isOpened():          # This ensures that the loop runs as long as the webcam is accessible and open
    ret, frame = cap.read()    # Read a frame from the webcam, 'ret' is a boolean that indicates if the frame was captured successfully.
    if not ret:                # If 'ret' is False, then the frame was not captured properly (e.g., end of stream or webcam issue)
        print("Failed to grab frame.")   # Print an error message if frame capture failed
        break                   # Exit the loop if frame capture fails

    # Convert the captured frame from BGR (color) to grayscale.
    # Converting to grayscale reduces the complexity of the image by removing color information.
    # Grayscale images are computationally easier to process because they only contain intensity values (brightness).
    # Most detection algorithms, like Haar cascades, work efficiently on grayscale images.
    # Using grayscale also speeds up object detection as it simplifies the image from 3 color channels (Red, Green, Blue) to just 1.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert BGR (color) frame to grayscale

    # Detect faces in the grayscale image using the pre-trained face cascade.
    # 'detectMultiScale()' is a method that detects objects (faces in this case) in an image.
    # Parameters:
    #   - gray: The grayscale image in which faces are to be detected.
    #   - 1.1: Scale factor used to compensate for faces appearing smaller or larger due to the scaling of the image.
    #   - 4: Minimum number of neighbors a rectangle should have to be considered a face.
    #     This is a trade-off between detecting faces and false positives; higher values reduce false positives.
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)  # Detect faces in the image with the specified parameters

    # Loop through all the faces detected in the image.
    # Each 'face' contains the coordinates and size of the detected face in the image.
    # The coordinates (x, y) represent the top-left corner of the face, and (w, h) are the width and height of the bounding box.
    for (x, y, w, h) in faces:        # 'x, y' are the coordinates of the top-left corner, 'w, h' are width and height of the face bounding box
        # Draw a rectangle around each detected face.
        # cv2.rectangle() is used to draw a rectangle on the image.
        # (255, 0, 0) represents the color of the rectangle (in BGR format: Blue, Green, Red).
        # 3 specifies the thickness of the rectangle in pixels.
        # Alternative: You could use a different color or no rectangle, depending on the visualization need.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw a blue rectangle around the face

        # Define a region of interest (ROI) within the detected face for further processing (eyes detection).
        # ROI allows us to focus detection efforts on a smaller portion of the image, which improves performance.
        # By detecting eyes within the face bounding box, we avoid scanning the entire image.
        roi_gray = gray[y:y + h, x:x + w]   # Crop the grayscale image to the detected face area (y:y+h, x:x+w defines the rectangle)
        roi_color = frame[y:y + h, x:x + w]  # Crop the color image to the detected face area (this is for drawing the eye bounding boxes)

        # Detect eyes in the face region using the pre-trained eye cascade.
        # Detecting eyes inside the face ROI speeds up the process compared to searching the entire image.
        # 'detectMultiScale()' is again used here with similar parameters as face detection, but on the cropped face region.
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes within the cropped face area (ROI)

        # Loop through each detected eye in the face region and draw a rectangle around it.
        # Similar to face detection, we draw a rectangle for each eye detected inside the face region.
        for (ex, ey, ew, eh) in eyes:         # 'ex, ey' are the coordinates of the top-left corner of the eye bounding box, 'ew, eh' are width and height
            # Draw a rectangle around each detected eye with the color (0, 0, 255) which is red in BGR format.
            # 3 specifies the thickness of the rectangle.
            # Alternative: You could draw circles instead of rectangles for eyes, or simply omit the drawing.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)  # Draw a red rectangle around each eye

    # Display the processed frame with face and eye detections
    # The 'cv2.imshow()' function displays the frame in a window.
    # This will continuously show the frame as the program processes each video frame from the webcam.
    # Alternative: You could save the processed video to a file using 'cv2.VideoWriter()' for post-processing or analysis.
    cv2.imshow('Face and Eye Detection', frame)  # Show the processed frame in a window named 'Face and Eye Detection'

    # Exit the loop if the user presses the 'q' key.
    # The 'cv2.waitKey(1)' waits for 1 millisecond and checks if any key was pressed.
    # If the 'q' key is pressed, 'ord('q')' returns the ASCII value for 'q', which exits the loop.
    # Alternative: Instead of waiting for the 'q' key, you could stop based on other conditions, such as a timer or specific frame count.
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 1 millisecond and check if the 'q' key is pressed
        break    # Exit the loop if the 'q' key is pressed (user presses 'q' to quit)

# After the loop ends, release the webcam and close all OpenCV windows.
# 'cap.release()' releases the video capture object and closes the webcam stream.
# 'cv2.destroyAllWindows()' closes all OpenCV windows that were opened during the program.
cap.release()        # Release the webcam and the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows (like the one displaying the video)
