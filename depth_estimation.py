import cv2
import torch
import numpy as np

print("Loading MiDaS model... Please wait...")

# Load MiDaS small model (fast and works on most PCs)
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform

print("Opening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map
    depth_map = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    # Near = brighter, Far = darker
    depth_map = 255 - depth_map

    # Apply color map for better visualization
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    cv2.imshow("Original Webcam", frame)
    cv2.imshow("Live Depth Estimation", depth_colored)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()