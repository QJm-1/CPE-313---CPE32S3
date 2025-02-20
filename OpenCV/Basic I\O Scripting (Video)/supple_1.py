import cv2

# Open the video file
cap = cv2.VideoCapture('test1.mp4')

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while True:
        ret, frame = cap.read()
        
        # If frame reading was successful, display it
        if not ret:
            break

        cv2.imshow('Video Playback', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()