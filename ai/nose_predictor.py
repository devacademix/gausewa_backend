# ai/nose_predictor.py
# UPDATED: Dual strategy
# 1. AI confidence >= 60% → cattle_id match
# 2. AI confidence < 60% → Deep neural embedding match (real-world photos ke liye)

import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os
import base64

class NosePredictor:
    def __init__(self):
        model_path = './ai/nose_model.keras'
        class_path = './ai/class_names.json'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model nahi mila: {model_path}')
        if not os.path.exists(class_path):
            raise FileNotFoundError(f'class_names.json nahi mila: {class_path}')

        print('Loading Nose Print AI model...')
        self.model = tf.keras.models.load_model(model_path)
        
        # Deep feature extractor - gets 256-D embedding
        self.feature_extractor = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('dense_1').output
        )

        with open(class_path) as f:
            self.class_names = json.load(f)

        print(f'Nose AI ready: {len(self.class_names)} cattle registered')
        print(f'Model input shape: {self.model.input_shape}')

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def get_image_hash(self, image_bytes: bytes) -> str:
        """
        Deep Neural hash — uses extracted features for robust matching
        """
        input_tensor = self.preprocess(image_bytes)
        features = self.feature_extractor.predict(input_tensor, verbose=0)[0]
        # Base64 encode the float32 array
        return base64.b64encode(features.tobytes()).decode('utf-8')

    def hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Do hashes kitne similar hain — 0.0 to 1.0 (Cosine Similarity)
        """
        try:
            f1 = np.frombuffer(base64.b64decode(hash1), dtype=np.float32)
            f2 = np.frombuffer(base64.b64decode(hash2), dtype=np.float32)
            
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Compute cosine similarity
            sim = float(np.dot(f1, f2) / (norm1 * norm2))
            
            # Since features are from ReLU, sim is [0, 1]. Returning directly.
            return sim
        except:
            return 0.0

    def identify(self, image_bytes: bytes) -> dict:
        """
        Photo se cattle identify karo.
        Returns: cow_id, confidence, match, top3, image_hash
        """
        input_tensor = self.preprocess(image_bytes)

        # AI prediction
        predictions  = self.model.predict(input_tensor, verbose=0)
        probs        = predictions[0]
        top_idx      = int(np.argmax(probs))
        confidence   = float(probs[top_idx])
        cow_id       = self.class_names[top_idx]

        # Top 3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {
                'cow_id':     self.class_names[int(i)],
                'confidence': round(float(probs[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        # Use neural embedding instead of a fragile average hash
        features = self.feature_extractor.predict(input_tensor, verbose=0)[0]
        img_hash = base64.b64encode(features.tobytes()).decode('utf-8')

        return {
            'cow_id':     cow_id,
            'confidence': round(confidence * 100, 2),
            'match':      confidence >= 0.60,   # 60% threshold
            'top3':       top3,
            'image_hash': img_hash,
            'all_probs':  [
                {'cattle_id': self.class_names[i], 'prob': round(float(probs[i]) * 100, 2)}
                for i in np.argsort(probs)[-10:][::-1]
            ]
        }

# Global singleton
predictor = NosePredictor()