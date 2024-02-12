# main.py

import cv2
import numpy as np
from line_detection_functions import region_of_interest, draw_lines, hough_lines, draw_middle_line

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Main loop to process each frame from the camera
while True:
    # Capture the frame
    ret, frame = cap.read()
    
    # Break the loop if there's no frame
    if not ret:
        break

    # Define the region of interest as a slightly larger square
    height, width = frame.shape[:2]
    roi_size = 70  # Increase the size of the ROI
    roi_vertices = np.array([[(width // 2 - roi_size, height // 2 - roi_size),
                              (width // 2 + roi_size, height // 2 - roi_size),
                              (width // 2 + roi_size, height // 2 + roi_size),
                              (width // 2 - roi_size, height // 2 + roi_size)]], dtype=np.int32)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Apply region of interest mask
    roi = region_of_interest(edges, roi_vertices)

    # Draw the ROI on the frame
    cv2.polylines(frame, roi_vertices, isClosed=True, color=(255, 0, 0), thickness=2)

    # Apply Hough Transform to detect lines
    lines = hough_lines(roi, rho=1, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=20)

    # Draw parallel lines on the original frame
    draw_lines(frame, lines)

    # Draw the middle line between parallel lines
    draw_middle_line(frame, lines)

    # Display the result
    cv2.imshow('Parallel Lines Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
