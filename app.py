import streamlit as st
import numpy as np
import torch
import cv2
import os
import requests
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

MODEL_PATH = "sam_model/sam_vit_b.pth"
URL = "https://drive.google.com/uc?export=download&id=1hlApkNA72sZpososng8hg_8cX_4MIrlP"

# Ensure the folder exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_sam():
    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

predictor = load_sam()
yolo_model = load_yolo()

st.title("🧠 Interactive Segmentation System (SAM + YOLO)")

# ----------------------------
# Session State (for points)
# ----------------------------
if "points" not in st.session_state:
    st.session_state.points = []

if "labels" not in st.session_state:
    st.session_state.labels = []

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Mode", ["Manual (Points)", "Manual (Box)", "Auto (YOLO)"])

    if mode == "Manual (Points)":
        label_type = st.radio("Point Type", ["Foreground", "Background"])

    if st.button("🧹 Clear Points"):
        st.session_state.points = []
        st.session_state.labels = []
        st.rerun()

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    predictor.set_image(image_np)

    # ----------------------------
    # Resize image for UI
    # ----------------------------
    max_width = 500
    h, w = image_np.shape[:2]
    scale = min(max_width / w, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image_np, (new_w, new_h))

    col1, col2 = st.columns([1.2, 1])

    # ----------------------------
    # Canvas (Input)
    # ----------------------------
    with col1:
        st.subheader("Input")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=Image.fromarray(resized_image),
            update_streamlit=True,
            height=new_h,
            width=new_w,
            drawing_mode="point" if "Points" in mode else "rect",
            key="canvas",
        )

    # ----------------------------
    # POINT MODE (FIXED)
    # ----------------------------
    if mode == "Manual (Points)" and canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]

        # Add ONLY new points
        if len(objects) > len(st.session_state.points):
            new_obj = objects[-1]

            x = int(new_obj["left"] / scale)
            y = int(new_obj["top"] / scale)

            label = 1 if label_type == "Foreground" else 0

            st.session_state.points.append([x, y])
            st.session_state.labels.append(label)

        if len(st.session_state.points) > 0 and st.button("Segment (Points)"):
            with st.spinner("Segmenting..."):
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(st.session_state.points),
                    point_labels=np.array(st.session_state.labels),
                    multimask_output=True,
                )

            best = np.argmax(scores)
            mask = masks[best]

            overlay = image_np.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0])

            # Draw points (correct colors)
            for (x, y), l in zip(st.session_state.points, st.session_state.labels):
                color = (0, 255, 0) if l == 1 else (255, 0, 0)
                cv2.circle(overlay, (x, y), 6, color, -1)

            with col2:
                st.subheader("Result")
                st.image(overlay)
                st.write(f"Confidence: {scores[best]:.3f}")

    # ----------------------------
    # BOX MODE
    # ----------------------------
    if mode == "Manual (Box)" and canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]

        if len(objects) > 0:
            rect = objects[-1]

            x1 = int(rect["left"] / scale)
            y1 = int(rect["top"] / scale)
            x2 = int((rect["left"] + rect["width"]) / scale)
            y2 = int((rect["top"] + rect["height"]) / scale)

            if st.button("Segment (Box)"):
                with st.spinner("Segmenting..."):
                    masks, scores, _ = predictor.predict(
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=True,
                    )

                best = np.argmax(scores)
                mask = masks[best]

                overlay = image_np.copy()
                overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0])

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

                with col2:
                    st.subheader("Result")
                    st.image(overlay)
                    st.write(f"Confidence: {scores[best]:.3f}")

    # ----------------------------
    # YOLO + SAM (AUTO)
    # ----------------------------
    if mode == "Auto (YOLO)" and st.button("Run YOLO + SAM"):
        with st.spinner("Detecting objects..."):
            results = yolo_model(image_np)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        class_names = yolo_model.names

        with col2:
            st.subheader("YOLO + SAM Results")

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                masks, scores, _ = predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True,
                )

                best = np.argmax(scores)
                mask = masks[best]

                # IoU calculation
                yolo_mask = np.zeros(mask.shape, dtype=bool)
                yolo_mask[y1:y2, x1:x2] = True

                intersection = np.logical_and(mask, yolo_mask).sum()
                union = np.logical_or(mask, yolo_mask).sum()
                iou = intersection / union if union != 0 else 0

                overlay = image_np.copy()
                overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 255])

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = class_names[int(classes[i])]
                conf = confidences[i]

                cv2.putText(
                    overlay,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                st.image(overlay, caption=f"{label} | IoU: {iou:.3f}")
