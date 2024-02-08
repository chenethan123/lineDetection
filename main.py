#Formula https://stackoverflow.com/questions/15780210/python-opencv-detect-parallel-lines#:~:text=Two%20lines%20y%20%3D%20k1%20*%20x,points%20that%20belong%20to%20line.
#Research on Hough Lines https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/


import cv2 as cv
import numpy as np

def detect_lines(frame):
    height, width = frame.shape[:2]

    # Define the region of interest (ROI)
    roi_top = int(height * 0.6)  # Top boundary of ROI (60% of the frame height)
    roi_bottom = height  # Bottom boundary of ROI (bottom of the frame)
    roi_left = int(width * 0.2)  # Left boundary of ROI (20% of the frame width)
    roi_right = int(width * 0.8)  # Right boundary of ROI (80% of the frame width)

    # Draw a rectangle to mark the ROI on the frame
    cv.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)  # Convert the ROI to grayscale

    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # Detect edges using Canny edge detection
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)  # Detect lines using Hough transform

    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # Get rho and theta values, which are the parameters of the line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Adjust line coordinates to match the original frame
            x1 += roi_left
            x2 += roi_left
            y1 += roi_top
            y2 += roi_top

            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the line on the frame

    return frame

def main():
    cap = cv.VideoCapture(0)  # Open the default camera (camera index 0)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        frame_with_lines = detect_lines(frame)

        # Resize the frame to fit the screen
        disp_img = cv.resize(frame_with_lines, (700, 800))

        cv.imshow('Line Detection', disp_img)  # Display the result

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
