import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import os  # To check the absolute path of the image file
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk  # To display images in Tkinter

#---------------------------------------------
# Function to segment the region of hand in the image
#---------------------------------------------
def segment(image, grayimage, threshold=75):
    # Apply threshold to create a binary image
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        # Find the largest contour, assuming it's the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# Function to count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(image, thresholded, segmented):
    # Find the convex hull of the segmented hand contour
    chull = cv2.convexHull(segmented)

    # Find extreme points (top, bottom, left, right) of the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # Calculate the center point (cX, cY) of the hand region
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # Calculate distances between the center and extreme points (left, right, top, bottom)
    distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_distance = distances[distances.argmax()]

    # Calculate the radius for the circular region of interest (ROI)
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Create a circular mask around the center point
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # Apply the circular mask to the thresholded image to focus on the hand region
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours in the circular region
    cnts, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    for i, c in enumerate(cnts):
        # Calculate bounding rectangle for each contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Count the contours that represent fingers (based on position and size)
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

# Function to find convexity defects (indentations between fingers)
def find_defects(contour):
    if len(contour) < 3:
        return None
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    return defects

# Function to load and process the image for finger count detection
def process_image(image_path, label_result):
    if not os.path.isfile(image_path):
        label_result.config(text="Error: Image not found!")
        return

    frame = cv2.imread(image_path)

    if frame is None:
        label_result.config(text="Error: Image not loaded.")
        return

    # Resize the image for better handling
    frame = imutils.resize(frame, width=700)
    clone = frame.copy()

    # Convert image to grayscale and apply Gaussian blur for noise reduction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Segment the hand region
    hand = segment(clone, gray)

    if hand is not None:
        (thresholded, segmented) = hand
        # Count the number of fingers in the hand region
        fingers = count(clone, thresholded, segmented)
        label_result.config(text=f"Fingers Count: {fingers}")
    else:
        label_result.config(text="Hand not detected.")

    # Convert the processed image to a format compatible with Tkinter
    img = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    # Update the image on the Tkinter panel
    panel.config(image=img)
    panel.image = img

# Function to open a file dialog and select an image
def upload_image(label_result):
    filename = filedialog.askopenfilename(initialdir=".", title="Select a File", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    if filename:
        # Process the selected image
        process_image(filename, label_result)

# Function for real-time hand gesture detection using the webcam
def hand_gesture_camera(label_result):
    # Start webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        label_result.config(text="Error: Unable to access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Mirror the video for better usability

        # Convert to HSV color space for better skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV space
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Threshold the image to extract skin color region
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Clean up the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Apply the mask to the original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the result to grayscale and apply thresholding
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 3:
                # Draw the convex hull around the hand
                hull = cv2.convexHull(contour)
                defects = find_defects(contour)

                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])

                        # Calculate the angle to determine fingertips
                        angle = np.arccos(np.dot(np.array(end) - np.array(far), np.array(start) - np.array(far)) /
                                          (np.linalg.norm(np.array(end) - np.array(far)) * np.linalg.norm(np.array(start) - np.array(far))))

                        if angle <= np.pi / 2:
                            finger_count += 1
                            cv2.circle(frame, far, 5, [0, 0, 255], -1)  # Mark the detected fingertips

                    # Display the number of fingers detected
                    cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Convert the processed frame to a format for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Update the image in the Tkinter window
        panel.config(image=img)
        panel.image = img

        # Display the processed frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the Tkinter GUI
root = Tk()
root.title("Hand Gestures Recognition")

# Add the title label at the top
title_label = Label(root, text="Hand Gestures Recognition", font=("Helvetica", 20, "bold"), fg="blue")
title_label.pack(pady=10)

# Label for displaying the result (finger count or error messages)
label_result = Label(root, text="Select an option to process.", font=("Helvetica", 14))
label_result.pack()

# Button to upload an image for processing
btn_upload = Button(root, text="Upload Image", font=("Helvetica", 12), command=lambda: upload_image(label_result))
btn_upload.pack(pady=10)

# Button to start real-time hand gesture recognition using the webcam
btn_camera = Button(root, text="Hand Gesture with Camera", font=("Helvetica", 12), command=lambda: hand_gesture_camera(label_result))
btn_camera.pack(pady=10)

# Label to display the processed image
panel = Label(root)
panel.pack()

# Start the Tkinter event loop
root.mainloop()
