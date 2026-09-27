import cv2

# Index 0 targets your built-in laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam active! Press 'q' on your keyboard to close the window.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live video window
    cv2.imshow("Step 1 - Camera Check", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()