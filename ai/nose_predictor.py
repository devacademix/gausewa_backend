import base64
import json
import logging
import os
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from ai.image_validation import ValidatedImage, validate_image_bytes


logger = logging.getLogger(__name__)


class NosePredictor:
    def __init__(self):
        model_path = "./ai/nose_model.keras"
        class_path = "./ai/class_names.json"
        model_info_path = "./ai/model_info.json"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model nahi mila: {model_path}")
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"class_names.json nahi mila: {class_path}")

        logger.info("Loading Nose Print AI model")
        self.model = tf.keras.models.load_model(model_path)
        self.feature_extractor = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self._resolve_embedding_layer().output,
        )

        with open(class_path, encoding="utf-8") as class_file:
            self.class_names = json.load(class_file)

        self.model_info = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, encoding="utf-8") as info_file:
                self.model_info = json.load(info_file)

        self.image_size = int(self.model_info.get("img_size", self.model.input_shape[1] or 224))
        self.confidence_threshold = float(self.model_info.get("confidence_threshold", 0.85))
        self.margin_threshold = float(self.model_info.get("margin_threshold", 0.15))
        self.max_entropy_ratio = float(self.model_info.get("max_entropy_ratio", 0.55))
        self.prototype_similarity_threshold = float(
            self.model_info.get("prototype_similarity_threshold", 0.20)
        )

        self.class_prototypes = self._build_class_prototypes()

        logger.info(
            "Nose AI ready: classes=%s input_shape=%s threshold=%.2f",
            len(self.class_names),
            self.model.input_shape,
            self.confidence_threshold,
        )

    def _resolve_embedding_layer(self) -> tf.keras.layers.Layer:
        try:
            return self.model.get_layer("dense_1")
        except ValueError:
            return self.model.layers[-2]

    def _build_class_prototypes(self) -> np.ndarray | None:
        classifier_layer = self.model.layers[-1]
        weights = classifier_layer.get_weights()
        if not weights:
            return None

        kernel = weights[0]
        if kernel.ndim != 2:
            return None

        prototypes = kernel.T.astype(np.float32)
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return prototypes / norms

    def extract_nose_roi(self, image: Image.Image) -> Image.Image:
        # Placeholder hook for a future dedicated cow nose detector / ROI extractor.
        return image

    def preprocess(self, validated_image: ValidatedImage) -> np.ndarray:
        image = self.extract_nose_roi(validated_image.image)
        image = image.convert("RGB").resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]

        return np.expand_dims(array, axis=0)

    def get_image_hash(self, image_bytes: bytes) -> str:
        validated_image = validate_image_bytes(image_bytes)
        input_tensor = self.preprocess(validated_image)
        features = self.feature_extractor.predict(input_tensor, verbose=0)[0].astype(np.float32)
        return base64.b64encode(features.tobytes()).decode("utf-8")

    def hash_similarity(self, hash1: str, hash2: str) -> float:
        try:
            feature_1 = np.frombuffer(base64.b64decode(hash1), dtype=np.float32)
            feature_2 = np.frombuffer(base64.b64decode(hash2), dtype=np.float32)

            norm_1 = np.linalg.norm(feature_1)
            norm_2 = np.linalg.norm(feature_2)
            if norm_1 == 0.0 or norm_2 == 0.0:
                return 0.0

            similarity = float(np.dot(feature_1, feature_2) / (norm_1 * norm_2))
            return max(0.0, min(1.0, similarity))
        except Exception:
            logger.exception("Failed to compute hash similarity")
            return 0.0

    def identify(self, image_bytes: bytes) -> dict[str, Any]:
        validated_image = validate_image_bytes(image_bytes)
        input_tensor = self.preprocess(validated_image)

        predictions = self.model.predict(input_tensor, verbose=0)[0].astype(np.float32)
        probabilities = np.clip(predictions, 1e-9, 1.0)
        probabilities = probabilities / probabilities.sum()

        features = self.feature_extractor.predict(input_tensor, verbose=0)[0].astype(np.float32)
        image_hash = base64.b64encode(features.tobytes()).decode("utf-8")

        top_indices = np.argsort(probabilities)[::-1]
        top_idx = int(top_indices[0])
        second_idx = int(top_indices[1]) if len(top_indices) > 1 else top_idx
        confidence = float(probabilities[top_idx])
        second_confidence = float(probabilities[second_idx]) if second_idx != top_idx else 0.0
        margin = max(0.0, confidence - second_confidence)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        entropy_ratio = entropy / np.log(len(probabilities))
        cow_id = self.class_names[top_idx]

        top3 = [
            {
                "cow_id": self.class_names[int(index)],
                "confidence": round(float(probabilities[index]) * 100, 2),
            }
            for index in top_indices[:3]
        ]

        prototype_similarity = self._calculate_prototype_similarity(features, top_idx)
        ood_analysis = self._build_ood_analysis(
            confidence=confidence,
            margin=margin,
            entropy_ratio=entropy_ratio,
            prototype_similarity=prototype_similarity,
        )

        logger.debug(
            "Prediction confidence=%.4f margin=%.4f entropy_ratio=%.4f prototype_similarity=%.4f",
            confidence,
            margin,
            entropy_ratio,
            prototype_similarity,
        )

        response = {
            "cow_id": cow_id,
            "confidence": round(confidence * 100, 2),
            "match": ood_analysis["is_valid"],
            "top3": top3,
            "image_hash": image_hash,
            "invalid_input": not ood_analysis["is_valid"],
            "message": "Invalid cow nose image" if not ood_analysis["is_valid"] else None,
            "rejection_reason": ood_analysis["reason"],
            "ood": {
                "confidence_threshold": round(self.confidence_threshold * 100, 2),
                "margin": round(margin * 100, 2),
                "entropy_ratio": round(entropy_ratio, 4),
                "prototype_similarity": round(prototype_similarity, 4),
                "signals": ood_analysis["signals"],
            },
            "all_probs": [
                {
                    "cattle_id": self.class_names[int(index)],
                    "prob": round(float(probabilities[index]) * 100, 2),
                }
                for index in top_indices[:10]
            ],
        }

        return response

    def _calculate_prototype_similarity(self, features: np.ndarray, class_index: int) -> float:
        if self.class_prototypes is None or class_index >= len(self.class_prototypes):
            return 0.0

        feature_norm = float(np.linalg.norm(features))
        if feature_norm == 0.0:
            return 0.0

        normalized_features = features / feature_norm
        return float(np.dot(normalized_features, self.class_prototypes[class_index]))

    def _build_ood_analysis(
        self,
        *,
        confidence: float,
        margin: float,
        entropy_ratio: float,
        prototype_similarity: float,
    ) -> dict[str, Any]:
        signals: list[str] = []

        if confidence < self.confidence_threshold:
            signals.append("low_confidence")
        if margin < self.margin_threshold:
            signals.append("low_margin")
        if entropy_ratio > self.max_entropy_ratio:
            signals.append("high_entropy")
        if prototype_similarity < self.prototype_similarity_threshold:
            signals.append("low_prototype_similarity")

        return {
            "is_valid": not signals,
            "reason": "Input is out of distribution for the cow nose model" if signals else None,
            "signals": signals,
        }


predictor = NosePredictor()
