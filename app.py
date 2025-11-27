# app.py - stable fallback-only mask generator (use this now)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import numpy as np
import uuid
import cv2

app = FastAPI()

# CORS - permissive for testing; restrict to your domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
MASK_DIR = Path("masks")
UPLOAD_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)


def fallback_mask_by_color(input_path: Path, mask_path: Path):
    """
    Heuristic segmentation that finds red/orange roof areas.
    Produces a mask PNG with white where roof is, transparent elsewhere.
    """
    im = cv2.imread(str(input_path))
    if im is None:
        raise RuntimeError("cv2 failed to read input image")

    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Threshold ranges for red/orange (two ranges for red wrap)
    lower1 = np.array([0, 35, 40])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 35, 40])
    upper2 = np.array([179, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Expand mask and then clean
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Prefer large connected components near top half
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = np.zeros_like(mask)
    if num_labels > 1:
        candidates = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            cy = centroids[i][1]
            # weight regions that are wide and nearer the top
            score = area * (1.0 + max(0, 1.0 - (cy / h)))
            candidates.append((score, i))
        if candidates:
            candidates.sort(reverse=True)
            top_i = candidates[0][1]
            best[labels == top_i] = 255
            # optional: include second if large enough
            if len(candidates) > 1 and candidates[1][0] > 0.35 * candidates[0][0]:
                second_i = candidates[1][1]
                best[labels == second_i] = 255
    else:
        best = mask

    # Smooth and remove small specks
    best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    best = cv2.morphologyEx(best, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Write as RGBA: white where mask, transparent elsewhere
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[best > 0] = [255,255,255,255]
    Image.fromarray(rgba).save(mask_path)


def mask_has_enough_pixels(mask_path: Path, threshold_pixels: int = 400):
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    return int((arr > 127).sum()) > threshold_pixels


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}.png"
    mask_path = MASK_DIR / f"{uid}_mask.png"

    # Save upload
    with input_path.open("wb") as f:
        f.write(await file.read())

    # Create mask using heuristic
    try:
        fallback_mask_by_color(input_path, mask_path)
    except Exception as e:
        return JSONResponse({"error": f"fallback mask generation failed: {e}"}, status_code=500)

    # Validate
    if not mask_has_enough_pixels(mask_path, threshold_pixels=400):
        return JSONResponse({"error": "mask not detected or too small"}, status_code=422)

    return JSONResponse({"mask_url": f"/masks/{mask_path.name}", "image_id": uid})


@app.get("/")
def home():
    return {"status": "ok", "service": "colorbond-segmentation-fallback"}


@app.get("/masks/{name}")
def get_mask(name: str):
    p = MASK_DIR / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)
