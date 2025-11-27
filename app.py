from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import subprocess
import uuid

app = FastAPI()

UPLOAD_DIR = Path("uploads")
MASK_DIR = Path("masks")
UPLOAD_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)


def run_rembg(input_path: Path, output_path: Path):
    subprocess.run(["rembg", "i", str(input_path), str(output_path)], check=True)


def clean_mask(mask_path: Path):
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


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{uid}.png"
    mask_path = MASK_DIR / f"{uid}_mask.png"

    with input_path.open("wb") as f:
        f.write(await file.read())

    try:
        run_rembg(input_path, mask_path)
        clean_mask(mask_path)
    except Exception as e:
        return JSONResponse({"error": f"mask generation failed: {e}"}, status_code=500)

    return JSONResponse({
        "mask_url": f"/masks/{mask_path.name}",
        "image_id": uid
    })


@app.get("/masks/{name}")
async def get_mask(name: str):
    p = MASK_DIR / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)
