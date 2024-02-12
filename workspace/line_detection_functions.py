# line_detection_functions.py

import cv2
import numpy as np

# Function to create a region of interest mask
def region_of_interest(img, vertices):
    # Create an empty mask with the same shape as the input image
    mask = np.zeros_like(img)
    
    # Fill the region of interest defined by vertices with white (255)
    cv2.fillPoly(mask, [vertices], 255)
    
    # Apply the mask to the input image using bitwise AND operation
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img

# Function to draw lines on the image
def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    # Check if lines are not None
    if lines is not None:
        # Iterate through each line in the list of lines
        for line in lines:
            # Unpack the line coordinates
            for x1, y1, x2, y2 in line:
                # Draw a line on the image using the specified color and thickness
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to apply Hough Transform and detect lines in the image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # Use OpenCV's HoughLinesP function to detect lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Function to draw a middle line between parallel lines
def draw_middle_line(img, lines, color=(0, 255, 0), thickness=3):
    # Check if lines are not None and the list is not empty
    if lines is not None and len(lines) > 0:
        # Calculate the average coordinates of the lines
        middle_line = np.mean(lines, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = middle_line.flatten()
        
        # Draw the middle line on the image using the specified color and thickness
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
