import cv2
import time

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

frame_id = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Save the resulting frame
        cv2.imwrite('path/to/save/images/frame{}.png'.format(frame_id), frame)
        frame_id += 1

        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Wait for 3 seconds
    time.sleep(3)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all the frames
cv2.destroyAllWindows()