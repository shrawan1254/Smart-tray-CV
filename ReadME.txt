# AI Smart Shelf (Grab-and-Go Checkout) 🛒🤖

A real-time, computer vision-based automated checkout system inspired by Amazon Go and smart cafeterias. Built with Python, OpenCV, and Ultralytics YOLOv8.

Instead of scanning barcodes, a camera monitors a designated "shelf" or tray. The AI tracks the initial inventory, detects when a user removes an item, and automatically calculates the bill in real-time.

##  Features
* **Zero-Touch Billing:** Automatically calculates the bill based on inventory depletion (Items Taken = Initial Stock - Current Stock).
* **Live On-Screen UI:** Displays a dynamic, semi-transparent receipt directly over the video feed.
* **Auto-Reset Checkout:** Hitting the Spacebar saves a formatted receipt to `receipt.txt`, flashes a success banner, and instantly resets the shelf baseline for the next customer.
* **Dynamic Pricing:** Prices and supported items are loaded from a simple `prices.json` file.
* **Edge Processing:** Runs entirely locally on CPU using the lightweight YOLOv8 Nano model—no cloud GPU required.

##  Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (Video capture, GUI, bounding boxes, ROI logic)
* **Object Detection:** Ultralytics YOLOv8 (yolov8n.pt)

##  How to Run Locally

### 1. Clone the repository and navigate to the directory:
```bash
git clone [https://github.com/YOUR_USERNAME/smart-shelf-checkout.git](https://github.com/YOUR_USERNAME/smart-shelf-checkout.git)
cd smart-shelf-checkout