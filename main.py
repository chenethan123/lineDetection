# Importing necessary libraries
import cv2
import numpy as np

# Function to create a region of interest mask
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Function to draw lines on the image
def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to apply Hough Transform and detect lines in the image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Function to draw a middle line between parallel lines
def draw_middle_line(img, lines, color=(0, 255, 0), thickness=3):
    if lines is not None and len(lines) > 0:
        middle_line = np.mean(lines, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = middle_line.flatten()
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

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
