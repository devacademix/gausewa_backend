import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_DIMENSION = 160
MAX_IMAGE_DIMENSION = 6000
MIN_ASPECT_RATIO = 0.75
MAX_ASPECT_RATIO = 1.33
MIN_SHARPNESS = 35.0
MIN_CONTRAST = 18.0
MIN_DYNAMIC_RANGE = 48.0
MIN_INFORMATIVE_TILE_RATIO = 0.55


class ImageValidationError(ValueError):
    def __init__(self, message: str, code: str = "invalid_image", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass
class ValidatedImage:
    image: Image.Image
    image_format: str
    width: int
    height: int
    aspect_ratio: float
    quality_metrics: dict[str, float]


def validate_upload_file(upload_file: Any) -> None:
    filename = (getattr(upload_file, "filename", "") or "").strip()
    content_type = (getattr(upload_file, "content_type", "") or "").lower().strip()
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            "Only JPG, JPEG, and PNG images are allowed",
            code="unsupported_file_type",
            details={"filename": filename, "allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
        )

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            "Invalid image content type",
            code="unsupported_content_type",
            details={"content_type": content_type, "allowed_content_types": sorted(ALLOWED_CONTENT_TYPES)},
        )


def validate_image_bytes(image_bytes: bytes) -> ValidatedImage:
    if not image_bytes:
        raise ImageValidationError("Image file is empty", code="empty_file")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            "Image file is too large",
            code="image_too_large",
            details={"max_bytes": MAX_IMAGE_BYTES},
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe_image:
            probe_image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("Uploaded file is not a valid image", code="corrupted_image") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            normalized_image = ImageOps.exif_transpose(source_image).convert("RGB")
            image_format = (source_image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Image data could not be decoded", code="invalid_image_bytes") from exc

    width, height = normalized_image.size
    aspect_ratio = width / height if height else 0.0

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            "Image resolution is too small",
            code="image_too_small",
            details={"min_dimension": MIN_IMAGE_DIMENSION, "width": width, "height": height},
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            "Image resolution is too large",
            code="image_too_large_dimensions",
            details={"max_dimension": MAX_IMAGE_DIMENSION, "width": width, "height": height},
        )

    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        raise ImageValidationError(
            "Image aspect ratio is invalid for cow nose identification",
            code="invalid_aspect_ratio",
            details={
                "aspect_ratio": round(aspect_ratio, 3),
                "min_aspect_ratio": MIN_ASPECT_RATIO,
                "max_aspect_ratio": MAX_ASPECT_RATIO,
            },
        )

    quality_metrics = calculate_image_quality_metrics(normalized_image)

    if quality_metrics["sharpness"] < MIN_SHARPNESS:
        raise ImageValidationError(
            "Image is too blurry for cow nose identification",
            code="image_too_blurry",
            details={"sharpness": round(quality_metrics["sharpness"], 2)},
        )

    if quality_metrics["contrast"] < MIN_CONTRAST or quality_metrics["dynamic_range"] < MIN_DYNAMIC_RANGE:
        raise ImageValidationError(
            "Image quality is too low for cow nose identification",
            code="low_quality_image",
            details={
                "contrast": round(quality_metrics["contrast"], 2),
                "dynamic_range": round(quality_metrics["dynamic_range"], 2),
            },
        )

    if quality_metrics["informative_tile_ratio"] < MIN_INFORMATIVE_TILE_RATIO:
        raise ImageValidationError(
            "Image does not appear to be a close-up cow nose photo",
            code="invalid_nose_region",
            details={"informative_tile_ratio": round(quality_metrics["informative_tile_ratio"], 2)},
        )

    return ValidatedImage(
        image=normalized_image.copy(),
        image_format=image_format,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        quality_metrics=quality_metrics,
    )


def calculate_image_quality_metrics(image: Image.Image) -> dict[str, float]:
    rgb_array = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    dynamic_range = float(gray.max() - gray.min())

    informative_tiles = 0
    total_tiles = 0
    for row in np.array_split(gray, 3, axis=0):
        for tile in np.array_split(row, 3, axis=1):
            total_tiles += 1
            tile_std = float(tile.std())
            tile_range = float(tile.max() - tile.min())
            if tile_std >= 12.0 and tile_range >= 40.0:
                informative_tiles += 1

    informative_tile_ratio = informative_tiles / total_tiles if total_tiles else 0.0

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "dynamic_range": dynamic_range,
        "informative_tile_ratio": informative_tile_ratio,
    }
