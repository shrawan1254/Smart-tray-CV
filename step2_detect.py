from datetime import datetime
import json
import os
import time
import cv2
from ultralytics import YOLO

# 1. Load Price Catalog from JSON
PRICES_FILE = "prices.json"
if not os.path.exists(PRICES_FILE):
    print(f"Error: {PRICES_FILE} not found. Ensure prices.json exists.")
    exit()

with open(PRICES_FILE, "r") as f:
    PRICE_CATALOG = json.load(f)

TRACKED_ITEM = "bottle"
UNIT_PRICE = PRICE_CATALOG.get(TRACKED_ITEM, 40.0)

# 2. Setup Model & Camera
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Request High Resolution (1280x720 or 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a resizable window
cv2.namedWindow("Smart Shelf - Grab and Go", cv2.WINDOW_NORMAL)

# Shelf tracking state variables
initial_count = 0
is_stocked = False
last_checkout_time = 0
last_checkout_bill = 0.0
last_checkout_qty = 0

print("\n--- GRAB-AND-GO SHELF (FULLSCREEN & AUTO-RESET) ---")
print("1. Place bottles inside the cyan shelf area.")
print("2. Press 's' to LOCK INITIAL STOCK.")
print("3. Take out bottles.")
print("4. Press SPACEBAR to CHECKOUT (Logs bill & auto-resets for next person).")
print("5. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define the shelf Region of Interest (left half of screen)
    shelf_x1, shelf_y1 = int(w * 0.05), int(h * 0.15)
    shelf_x2, shelf_y2 = int(w * 0.65), int(h * 0.85)

    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)[0]
    current_tray_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]

        if label == TRACKED_ITEM:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if item center is within the shelf boundary
            if (shelf_x1 <= cx <= shelf_x2) and (shelf_y1 <= cy <= shelf_y2):
                current_tray_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"On Shelf: {label}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # Draw Shelf Boundary Box
    shelf_color = (0, 255, 255) if is_stocked else (100, 100, 100)
    cv2.rectangle(
        frame, (shelf_x1, shelf_y1), (shelf_x2, shelf_y2), shelf_color, 2
    )

    status_tag = (
        f"SHELF: ACTIVE ({current_tray_count} on shelf)"
        if is_stocked
        else "SHELF: NOT LOCKED (Press 's' to stock)"
    )
    cv2.putText(
        frame,
        status_tag,
        (shelf_x1 + 10, shelf_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        shelf_color,
        2,
    )

    # Calculate quantities and bill
    taken_count = 0
    total_bill = 0.0
    if is_stocked:
        taken_count = max(0, initial_count - current_tray_count)
        total_bill = taken_count * UNIT_PRICE

    # Draw Semi-Transparent Receipt Panel on Top-Right
    panel_w = 360
    panel_h = 340
    panel_x1 = w - panel_w - 20
    panel_y1 = 20
    panel_x2 = w - 20
    panel_y2 = panel_y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (25, 25, 25), -1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        "--- GRAB & GO CART ---",
        (panel_x1 + 25, panel_y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    if not is_stocked:
        cv2.putText(
            frame,
            "Place bottles inside",
            (panel_x1 + 20, panel_y1 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            1,
        )
        cv2.putText(
            frame,
            "and press 's' to start",
            (panel_x1 + 20, panel_y1 + 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            1,
        )
    else:
        cv2.putText(
            frame,
            f"Initial Stock: {initial_count} items",
            (panel_x1 + 20, panel_y1 + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"Taken: {taken_count} x Rs. {UNIT_PRICE:.2f}",
            (panel_x1 + 20, panel_y1 + 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 120),
            2,
        )
        cv2.putText(
            frame,
            "[SPACE] Checkout & Reset",
            (panel_x1 + 20, panel_y1 + 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            1,
        )

        cv2.line(
            frame,
            (panel_x1 + 15, panel_y2 - 60),
            (panel_x2 - 15, panel_y2 - 60),
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"BILL: Rs. {total_bill:.2f}",
            (panel_x1 + 20, panel_y2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
        )

    # Display on-screen checkout success banner for 3 seconds
    if time.time() - last_checkout_time < 3.0:
        banner = frame.copy()
        cv2.rectangle(banner, (0, h - 80), (w, h), (0, 150, 0), -1)
        cv2.addWeighted(banner, 0.85, frame, 0.15, 0, frame)
        cv2.putText(
            frame,
            f"PAID: Rs. {last_checkout_bill:.2f} for {last_checkout_qty} item(s). Ready for next customer!",
            (30, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Smart Shelf - Grab and Go", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        initial_count = current_tray_count
        is_stocked = True
        print(f"[LOCKED] Initial stock set to: {initial_count} {TRACKED_ITEM}s.")
    
    # Check Out & Auto-Reset Logic
    elif key == 32:  # Spacebar to finalize checkout
        if is_stocked and taken_count > 0:
            now_str = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
            receipt_entry = f"""========================================
       SMART GRAB-AND-GO STORE         
========================================
Date/Time : {now_str}
Status    : Paid / Completed
----------------------------------------
Item      : {TRACKED_ITEM.capitalize()}
Quantity  : {taken_count}
Rate      : Rs. {UNIT_PRICE:.2f}
----------------------------------------
TOTAL PAID: Rs. {total_bill:.2f}
========================================

"""
            # Append transaction silently to receipt.txt
            with open("receipt.txt", "a") as f:
                f.write(receipt_entry)

            print(receipt_entry)
            print(f"[LOGGED] Transaction recorded in receipt.txt.")

            # Trigger banner
            last_checkout_time = time.time()
            last_checkout_bill = total_bill
            last_checkout_qty = taken_count
            
            # --- AUTO-RESET ---
            # Set the new baseline stock to whatever is currently left on the shelf
            initial_count = current_tray_count
            print(f"[READY] Shelf auto-reset. New stock baseline: {initial_count}")

        elif is_stocked and taken_count == 0:
            print("[INFO] No items removed from the shelf to bill.")

cap.release()
cv2.destroyAllWindows()