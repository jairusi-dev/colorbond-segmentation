# app.py (DROP-IN replacement)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import subprocess
import uuid
import cv2
import io

app = FastAPI()

# CORS - keep permissive for testing; lock down in production
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

def run_rembg(input_path: Path, output_path: Path):
    """Attempt rembg CLI. May fail if rembg or system libs missing."""
    subprocess.run(["rembg", "i", str(input_path), str(output_path)], check=True)

def clean_mask(mask_path: Path):
    """Normalize mask to hard white/transparent with morphology."""
    img = Image.open(mask_path).convert("RGBA")
    alpha = img.split()[-1]
    arr = np.array(alpha)
    arr = (arr > 30).astype(np.uint8) * 255
    img = Image.fromarray(arr).convert("L")
    img = img.filter(ImageFilter.MaxFilter(7))
    img = img.filter(ImageFilter.MinFilter(5))

    out = Image.new("RGBA", img.size, (0,0,0,0))
    arr2 = np.array(img)
    final = np.zeros((arr2.shape[0], arr2.shape[1], 4), dtype=np.uint8)
    final[arr2 > 127] = [255,255,255,255]
    Image.fromarray(final).save(mask_path)

def fallback_mask_by_color(input_path: Path, mask_path: Path):
    """
    Simple heuristic fallback:
    - Convert to HSV
    - Threshold for red/orange hues (common for roofs)
    - Clean with morphology and choose large regions near top half
    """
    im = cv2.imread(str(input_path))
    if im is None:
        raise RuntimeError("cv2 failed to read input image")

    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Threshold ranges for red/orange (two ranges for red wrap)
    lower1 = np.array([0, 50, 40])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 50, 40])
    upper2 = np.array([179, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Expand mask and then clean
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Heuristic: prefer big connected components near top half
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = np.zeros_like(mask)
    if num_labels > 1:
        candidates = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            cy = centroids[i][1]
            # weight regions that are wider and more toward top
            score = area * (1.0 + max(0, 1.0 - (cy / h)))
            candidates.append((score, i))
        if candidates:
            candidates.sort(reverse=True)
            # take top region(s). sum top 1-2
            top_i = candidates[0][1]
            best[labels == top_i] = 255
            if len(candidates) > 1 and candidates[1][0] > 0.4 * candidates[0][0]:
                second_i = candidates[1][1]
                best[labels == second_i] = 255
    else:
        best = mask

    # final morphology to smooth
    best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    best = cv2.morphologyEx(best, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # write as RGBA white where mask, transparent elsewhere
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[best > 0] = [255,255,255,255]
    Image.fromarray(rgba).save(mask_path)

def mask_has_enough_pixels(mask_path: Path, threshold_pixels:int=1000) -> bool:
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    return int((arr>127).sum()) > threshold_pixels

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}.png"
    mask_path = MASK_DIR / f"{uid}_mask.png"

    with input_path.open("wb") as f:
        f.write(await file.read())

    # Try rembg first; if it fails, fallback to color heuristic
    rembg_failed = False
    try:
        run_rembg(input_path, mask_path)
        clean_mask(mask_path)
        if not mask_has_enough_pixels(mask_path, threshold_pixels=500):
            # rembg produced too small a mask; treat as failure
            rembg_failed = True
    except Exception:
        rembg_failed = True

    if rembg_failed:
        try:
            fallback_mask_by_color(input_path, mask_path)
            clean_mask(mask_path)
        except Exception as e:
            return JSONResponse({"error": f"mask generation failed (fallback error): {e}"}, status_code=500)

    # Final check
    if not mask_has_enough_pixels(mask_path, threshold_pixels=500):
        return JSONResponse({"error":"mask not detected or too small"}, status_code=422)

    return JSONResponse({
        "mask_url": f"/masks/{mask_path.name}",
        "image_id": uid
    })


@app.get("/")
def home():
    return {"status":"ok","service":"colorbond-segmentation"}


@app.get("/masks/{name}")
async def get_mask(name: str):
    p = MASK_DIR / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)
