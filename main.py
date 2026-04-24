import base64
import logging
import os
import uuid
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ai.image_validation import ImageValidationError, validate_upload_file
from ai.nose_predictor import predictor
from config import JWT_EXPIRE, JWT_SECRET, cows_col


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Gau Sewa MVP API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


def create_token(user_id: str, role: str = "farmer") -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=["HS256"],
        )
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def build_error_detail(message: str, code: str, **extra) -> dict:
    detail = {
        "success": False,
        "error": code,
        "message": message,
    }
    detail.update({key: value for key, value in extra.items() if value is not None})
    return detail


async def validate_and_predict_image(nose_photo: UploadFile) -> tuple[bytes, dict]:
    logger.debug(
        "Image received filename=%s content_type=%s",
        nose_photo.filename,
        nose_photo.content_type,
    )

    try:
        validate_upload_file(nose_photo)
        image_bytes = await nose_photo.read()
        result = predictor.identify(image_bytes)
    except ImageValidationError as exc:
        logger.warning("Validation failure code=%s details=%s", exc.code, exc.details)
        raise HTTPException(
            status_code=400,
            detail=build_error_detail(exc.message, exc.code, details=exc.details),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected identify pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=build_error_detail(
                "Internal server error during cow identification",
                "identify_pipeline_error",
            ),
        ) from exc

    if result.get("invalid_input"):
        logger.warning(
            "Prediction rejected confidence=%s signals=%s",
            result.get("confidence"),
            result.get("ood", {}).get("signals", []),
        )
        raise HTTPException(
            status_code=400,
            detail=build_error_detail(
                result.get("message") or "Invalid cow nose image",
                "invalid_cow_nose_image",
                confidence=result.get("confidence"),
                rejection_reason=result.get("rejection_reason"),
                ood=result.get("ood"),
            ),
        )

    logger.debug("Prediction accepted confidence=%s", result.get("confidence"))
    return image_bytes, result


@app.post("/api/auth/login")
async def login(phone: str, otp: str):
    if len(otp) != 4 or not otp.isdigit():
        raise HTTPException(400, "OTP 4 digits ka hona chahiye")
    token = create_token(user_id=phone, role="farmer")
    return {"token": token, "expires_in": JWT_EXPIRE * 60}


@app.post("/api/cows/register")
async def register_cow(
    nose_photo: UploadFile = File(...),
    owner_name: str = Form("Farmer"),
    owner_phone: str = Form(""),
    breed: str = Form("Gir"),
    age_years: float = Form(3.0),
    village: str = Form(""),
    district: str = Form("", alias="District"),
    state: str = Form("UP"),
):
    image_bytes, result = await validate_and_predict_image(nose_photo)
    img_hash = result["image_hash"]

    all_cows = list(
        cows_col.find(
            {"image_hash": {"$exists": True}},
            {"_id": 0, "image_hash": 1, "cow_id": 1, "owner_name": 1},
        )
    )

    for existing_cow in all_cows:
        similarity = predictor.hash_similarity(
            img_hash,
            existing_cow.get("image_hash", ""),
        )
        if similarity > 0.85:
            raise HTTPException(
                400,
                f"Yeh gaay pehle se registered hai! "
                f"ID: {existing_cow['cow_id']} | "
                f"Maalik: {existing_cow['owner_name']} | "
                f"Photo similarity: {similarity:.0%}",
            )

    if result["match"]:
        existing = cows_col.find_one({"cattle_id": result["cow_id"]})
        if existing:
            raise HTTPException(
                400,
                f"Yeh gaay pehle se registered hai! "
                f"ID: {existing['cow_id']} | "
                f"Maalik: {existing['owner_name']}",
            )

    cow_id = f"COW-{str(uuid.uuid4())[:8].upper()}"
    photo_b64 = base64.b64encode(image_bytes).decode()

    cow_doc = {
        "cow_id": cow_id,
        "cattle_id": result["cow_id"],
        "ai_confidence": result["confidence"],
        "image_hash": img_hash,
        "owner_name": owner_name,
        "owner_phone": owner_phone,
        "breed": breed,
        "age_years": age_years,
        "village": village,
        "district": district,
        "state": state,
        "nose_photo": photo_b64,
        "registered_at": datetime.utcnow().isoformat(),
        "health_status": "GREEN",
        "stolen": False,
        "vaccinations": [],
    }

    cows_col.insert_one(cow_doc)

    return {
        "success": True,
        "cow_id": cow_id,
        "cattle_id": result["cow_id"],
        "confidence": result["confidence"],
        "message": f"Gaay successfully registered: {cow_id}",
    }


@app.post("/api/cows/identify")
async def identify_cow(nose_photo: UploadFile = File(...)):
    _, result = await validate_and_predict_image(nose_photo)
    img_hash = result["image_hash"]

    all_cows = list(
        cows_col.find(
            {"image_hash": {"$exists": True}},
            {"_id": 0, "nose_photo": 0},
        )
    )

    best_match = None
    best_similarity = 0.0

    for cow in all_cows:
        similarity = predictor.hash_similarity(
            img_hash,
            cow.get("image_hash", ""),
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cow

    if best_match and best_similarity >= 0.75:
        return {
            "found": True,
            "confidence": round(best_similarity * 100, 2),
            "model_confidence": result["confidence"],
            "match_method": "photo_hash",
            "cow": best_match,
        }

    if result["match"]:
        cow = cows_col.find_one(
            {"cattle_id": result["cow_id"]},
            {"_id": 0, "nose_photo": 0},
        )
        if cow:
            return {
                "found": True,
                "confidence": result["confidence"],
                "match_method": "ai_model",
                "cow": cow,
            }

    return {
        "found": False,
        "confidence": result["confidence"],
        "message": "Gaay system mein registered nahi hai - pehle register karo",
        "ai_top3": result.get("top3", []),
        "best_hash_match": round(best_similarity * 100, 2) if best_match else 0,
    }


@app.get("/api/cows/{cow_id}")
async def get_cow_profile(cow_id: str):
    cow = cows_col.find_one(
        {"cow_id": cow_id},
        {"_id": 0, "nose_photo": 0},
    )
    if not cow:
        raise HTTPException(404, "Gaay nahi mili")
    return cow


@app.post("/api/cows/{cow_id}/report-stolen")
async def report_stolen(cow_id: str):
    result = cows_col.update_one(
        {"cow_id": cow_id},
        {"$set": {"stolen": True, "stolen_at": datetime.utcnow().isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Gaay nahi mili")
    return {"success": True, "message": "Chori ki report darj ho gayi"}


@app.post("/api/cows/{cow_id}/vaccinate")
async def add_vaccination(
    cow_id: str,
    vaccine: str = "FMD",
    date: str = "",
    next_due: str = "",
):
    vax = {
        "vaccine": vaccine,
        "date": date or datetime.utcnow().strftime("%Y-%m-%d"),
        "next_due": next_due,
    }
    result = cows_col.update_one(
        {"cow_id": cow_id},
        {"$push": {"vaccinations": vax}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Gaay nahi mili")
    return {"success": True, "vaccination": vax}


@app.get("/api/cows")
async def list_cows(state: str = None, district: str = None):
    query = {}
    if state:
        query["state"] = state
    if district:
        query["district"] = district
    cows = list(cows_col.find(query, {"_id": 0, "nose_photo": 0}).limit(100))
    return {"cows": cows, "count": len(cows)}


@app.get("/api/dashboard")
async def dashboard():
    total = cows_col.count_documents({})
    stolen = cows_col.count_documents({"stolen": True})
    by_state = list(
        cows_col.aggregate(
            [{"$group": {"_id": "$state", "count": {"$sum": 1}}}]
        )
    )
    return {"total_cows": total, "stolen_cows": stolen, "by_state": by_state}


@app.get("/")
async def health_check():
    return {"status": "running", "api": "Gau Sewa MVP", "version": "4.0"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
