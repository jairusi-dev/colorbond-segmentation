# app.py — Gemini mask generator (no rembg, no heavy libs)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
from io import BytesIO
import uuid
import base64
import os

from google import genai

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# CORS (keep * for testing; restrict later)
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


def generate_roof_mask_with_gemini(image_bytes: bytes):
    """
    Ask Gemini for segmentation including the roof mask.
    """
    resp = client.images.segment(
        images=[image_bytes],
        model="gemini-2.0-flash",  # free-tier compatible image model
        instructions="Return only the mask for the ROOF area as a base64 PNG."
    )

    # Convert to dict
    data = resp.to_dict()

    # Look for any mask
    masks = data.get("masks", [])
    if not masks:
        raise RuntimeError("Gemini returned no masks")

    # Take the first mask (roof mask based on instructions)
    mask_b64 = masks[0].get("mask")
    if not mask_b64:
        raise RuntimeError("Gemini returned mask object, but no base64 mask found")

    # Decode and return mask image
    mask_png = base64.b64decode(mask_b64)
    return mask_png


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}.png"
    mask_path = MASK_DIR / f"{uid}_mask.png"

    # Save upload
    content = await file.read()
    with input_path.open("wb") as f:
        f.write(content)

    try:
        mask_png_bytes = generate_roof_mask_with_gemini(content)
        with mask_path.open("wb") as f:
            f.write(mask_png_bytes)
    except Exception as e:
        return JSONResponse({"error": f"Gemini mask generation failed: {e}"}, status_code=500)

    return {"mask_url": f"/masks/{mask_path.name}", "image_id": uid}


@app.get("/")
def home():
    return {"status":"ok", "service":"colorbond-segmentation-gemini"}


@app.get("/masks/{name}")
def get_mask(name: str):
    p = MASK_DIR / name
    if not p.exists():
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(p)
