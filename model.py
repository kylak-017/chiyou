import json
import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _to_rgb_float01(image: np.ndarray) -> np.ndarray:
    """
    Convert an image-like array to RGB float32 in [0, 1].

    Supported shapes:
    - (H, W) grayscale
    - (H, W, 1) grayscale
    - (H, W, 3) RGB/BGR (assumed RGB)
    - (H, W, 4) RGBA (alpha dropped)
    - (3, H, W) channel-first RGB
    - (4, H, W) channel-first RGBA

    Note: This function does NOT load image files (PNG/JPG). Pass a NumPy array.
    """
    x = np.asarray(image)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        # channel-first -> channel-last
        x = np.transpose(x, (1, 2, 0))

    if x.ndim != 3 or x.shape[-1] not in (3, 4):
        raise ValueError(f"Expected image with shape (H,W,3/4) or (3/4,H,W); got {x.shape}")

    if x.shape[-1] == 4:
        x = x[..., :3]

    x = x.astype(np.float32, copy=False)
    if np.nanmax(x) > 1.0:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x


def image_to_numpy(image) -> np.ndarray:
    """
    Convert common "image" inputs into an RGB NumPy array (H, W, 3) uint8 (0..255).

    Supported inputs:
    - **NumPy array**: returned as-is (after shape normalization to RGB)
    - **File path** (str): requires Pillow (PIL). Loads image via PIL and converts to RGB.
    - **PIL Image**: requires Pillow (PIL). Converts to RGB and returns array.

    This function is intentionally small and dependency-light. If Pillow isn't installed,
    file-path / PIL inputs will raise a clear error.
    """
    if isinstance(image, np.ndarray):
        x01 = _to_rgb_float01(image)  # float32 0..1
        return (x01 * 255.0 + 0.5).astype(np.uint8)

    # Try Pillow if present (optional dependency)
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "To pass a file path or PIL Image, install Pillow (pip install pillow). "
            "Otherwise pass a NumPy array of shape (H,W,3)."
        ) from e

    if isinstance(image, str):
        im = Image.open(image).convert("RGB")
        return np.asarray(im, dtype=np.uint8)

    if isinstance(image, Image.Image):
        im = image.convert("RGB")
        return np.asarray(im, dtype=np.uint8)

    raise TypeError(f"Unsupported image type: {type(image)}")


def _maybe_downsample(x: np.ndarray, max_pixels: int) -> np.ndarray:
    """Cheap stride-based downsample to cap compute. Keeps aspect ratio-ish."""
    if max_pixels is None:
        return x
    h, w = x.shape[:2]
    if h * w <= max_pixels:
        return x
    stride = int(np.ceil(np.sqrt((h * w) / max_pixels)))
    return x[::stride, ::stride, :]


COLOR_LIST = [
    "red",
    "orange",
    "magenta",
    "black",
    "indigo",
    "purple",
    "gold",
    "pink",
    "brown",
    "blue",
    "silver",
    "yellow",
    "green",
    "turquoise",
    "white",
    "grey",
]


DEFAULT_COLOR_PROTOTYPES_RGB01 = {
    # Prototypes are in RGB, scaled to [0, 1]. These are tunable.
    "red": (1.00, 0.00, 0.00),
    "orange": (1.00, 0.55, 0.00),
    "magenta": (1.00, 0.00, 1.00),
    "black": (0.00, 0.00, 0.00),
    "indigo": (0.29, 0.00, 0.51),
    "purple": (0.50, 0.00, 0.50),
    "gold": (1.00, 0.84, 0.00),
    "pink": (1.00, 0.75, 0.80),
    "brown": (0.65, 0.16, 0.16),
    "blue": (0.00, 0.00, 1.00),
    "silver": (0.75, 0.75, 0.75),
    "yellow": (1.00, 1.00, 0.00),
    "green": (0.00, 0.50, 0.00),
    "turquoise": (0.25, 0.88, 0.82),
    "white": (1.00, 1.00, 1.00),
    "grey": (0.50, 0.50, 0.50),
}


class PrototypeColorCNN:
    """
    A tiny CNN-like model (1x1 convolution) that outputs a distribution over named colors.

    How it works:
    - Each pixel is scored against each color prototype using logits proportional to:
        -||x - p||^2 / temperature
      The pixel-wise softmax gives a soft assignment to each color.
    - We average pixel probabilities to get image-level percentages.

    This is a valid convolutional (CNN) classifier, but it uses fixed prototypes (no training needed).
    """

    def __init__(
        self,
        color_list: list[str] = None,
        prototypes_rgb01: dict[str, tuple[float, float, float]] = None,
        temperature: float = 0.02,
        max_pixels: int = 200_000,
    ):
        self.color_list = list(color_list) if color_list is not None else list(COLOR_LIST)
        self.prototypes_rgb01 = prototypes_rgb01 or dict(DEFAULT_COLOR_PROTOTYPES_RGB01)
        self.temperature = float(temperature)
        self.max_pixels = max_pixels

        missing = [c for c in self.color_list if c not in self.prototypes_rgb01]
        if missing:
            raise ValueError(f"Missing prototypes for: {missing}")
        if not (self.temperature > 0):
            raise ValueError("temperature must be > 0")

        P = np.array([self.prototypes_rgb01[c] for c in self.color_list], dtype=np.float32)  # (C,3)
        # For logits = (2 x·p - ||p||^2) / T  (the -||x||^2 term cancels across classes in softmax)
        self.W = (2.0 * P / self.temperature).astype(np.float32)  # (C,3)
        self.b = (-(P * P).sum(axis=1) / self.temperature).astype(np.float32)  # (C,)

    def predict_percentages(self, image: np.ndarray) -> dict[str, float]:
        """
        Returns dict of {color_name: percentage} summing to ~100.
        """
        x = _to_rgb_float01(image)
        x = _maybe_downsample(x, self.max_pixels)

        # 1x1 conv over pixels: logits[h,w,c] = sum_k x[h,w,k]*W[c,k] + b[c]
        logits = x @ self.W.T + self.b  # (H,W,C)
        probs = _softmax(logits, axis=-1)  # (H,W,C)
        mean_probs = probs.mean(axis=(0, 1))  # (C,)
        pct = (mean_probs * 100.0).astype(np.float64)
        return {c: float(p) for c, p in zip(self.color_list, pct)}

    def predict_percentages_array(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a NumPy array of shape (C,) with percentages in **self.color_list** order.
        """
        x = _to_rgb_float01(image)
        x = _maybe_downsample(x, self.max_pixels)
        logits = x @ self.W.T + self.b  # (H,W,C)
        probs = _softmax(logits, axis=-1)  # (H,W,C)
        mean_probs = probs.mean(axis=(0, 1))  # (C,)
        return (mean_probs * 100.0).astype(np.float64)


def extract_color_percentages(image: np.ndarray) -> dict[str, float]:
    """
    Convenience wrapper using the default PrototypeColorCNN configuration.
    """
    return PrototypeColorCNN().predict_percentages(image)


def extract_color_percentages_array(image) -> np.ndarray:
    """
    Input: an image (file path / PIL image / NumPy array)
    Output: a NumPy array of shape (16,) with percentages matching COLOR_LIST order.

    Use COLOR_LIST to interpret indices, e.g. idx 0 -> 'red', idx 1 -> 'orange', ...
    """
    rgb_uint8 = image_to_numpy(image)
    return PrototypeColorCNN().predict_percentages_array(rgb_uint8)




class ScentRecommender:
    def __init__(self, color_emotion_data, va_lookup):
        """
        color_emotion_data: Your grouped JSON (e.g., {"red": ["assertive", "calm"]})
        va_lookup: Dictionary mapping descriptors to (Valence, Arousal) tuples
        """
        self.color_emotion_data = color_emotion_data
        self.va_lookup = va_lookup
        self.color_profiles = self._generate_color_va_profiles()

    def _generate_color_va_profiles(self):
        """Calculates the mean V-A for every color in your dataset."""
        profiles = {}
        for color, descriptors in self.color_emotion_data.items():
            v_scores = []
            a_scores = []
            
            for desc in descriptors:
                if desc in self.va_lookup:
                    v, a = self.va_lookup[desc]
                    v_scores.append(v)
                    a_scores.append(a)
            
            if v_scores:
                profiles[color] = {
                    "v": sum(v_scores) / len(v_scores),
                    "a": sum(a_scores) / len(a_scores)
                }
        return profiles

    def get_scent(self, detected_colors=None, color_percentages: dict | None = None):
        """
        Provide either:
        - detected_colors: List of colors found in image (e.g., ['red', 'red', 'blue'])
        - color_percentages: Dict like {"red": 80.0, "blue": 20.0} (preferred)
        """
        if color_percentages is None and detected_colors is None:
            raise ValueError("Provide detected_colors or color_percentages")

        total_v = 0.0
        total_a = 0.0
        total_w = 0.0

        if color_percentages is not None:
            for color, pct in color_percentages.items():
                if color in self.color_profiles:
                    w = float(pct) / 100.0
                    total_v += self.color_profiles[color]["v"] * w
                    total_a += self.color_profiles[color]["a"] * w
                    total_w += w
        else:
            for color in detected_colors:
                if color in self.color_profiles:
                        total_v += self.color_profiles[color]["v"]
                        total_a += self.color_profiles[color]["a"]
                        total_w += 1.0

        # Calculate the final Image Average
        final_v = (total_v / total_w) if total_w else 0.0
        final_a = (total_a / total_w) if total_w else 0.0

        # Scent Logic based on Valence and Arousal
        # High Arousal (> 0.5), Low Valence (< 0) -> Eaglewood/Oud (Intense/Sharp)
        # Low Arousal (< 0), High Valence (> 0.5) -> Chamomile (Calm/Soothing)
        
        if final_a > 0.5:
            if final_v < 0:
                scent = "Eaglewood (Oud) - Intense & Assertive"
            else:
                scent = "Bergamot - Bright & Energizing"
        elif final_a < 0:
            if final_v > 0.4:
                scent = "Chamomile - Peaceful & Pure"
            else:
                scent = "Sandalwood - Deep & Reflective"
        else:
            scent = "White Musk - Balanced & Neutral"

        return {
            "coordinates": (round(final_v, 2), round(final_a, 2)),
            "recommendation": scent
        }

# --- EXAMPLE USAGE ---

# 1. Your Descriptor Map
va_descriptors = {
    # High Arousal, Positive Valence (Excited, Happy)
    "active": (0.5, 0.9),
    "adventurous": (0.8, 0.7),
    "ambitious": (0.5, 0.6),
    "artistic": (0.6, 0.3),
    "assertive": (0.3, 0.8),
    "attention-getting": (-0.2, 0.4),
    "cheerful": (0.9, 0.6),
    "confident": (0.7, 0.5),
    "creative": (0.7, 0.4),
    "determined": (0.4, 0.7),
    "energising and stimulating": (0.8, 0.9),
    "enthusiastic": (0.9, 0.8),
    "exciting": (0.9, 0.9),
    "exhilarating": (0.9, 0.8),
    "flamboyant": (0.6, 0.7),
    "happy": (1.0, 0.6),
    "hopeful": (0.8, 0.3),
    "imaginative": (0.7, 0.4),
    "innovative": (0.7, 0.5),
    "inspiring/inspirative": (0.9, 0.7),
    "motivating": (0.8, 0.7),
    "optimistic": (0.9, 0.5),
    "passionate": (0.8, 0.8),
    "positive": (0.8, 0.4),
    "rejuvenating": (0.8, 0.2),
    "spontaneous": (0.6, 0.7),
    "uplifting": (0.9, 0.6),
    "winning": (1.0, 0.8),

    # High Arousal, Negative Valence (Angry, Anxious)
    "aggressive": (-0.8, 0.8),
    "angry and quick-tempered": (-0.9, 0.9),
    "anxiety producing/anxious": (-0.7, 0.8),
    "bigoted": (-0.9, 0.5),
    "bossy": (-0.5, 0.6),
    "brutal": (-1.0, 0.9),
    "challenging": (-0.2, 0.6),
    "cynical": (-0.6, 0.2),
    "demanding": (-0.5, 0.7),
    "deceptive": (-0.8, 0.4),
    "domineering": (-0.7, 0.7),
    "envious": (-0.7, 0.5),
    "fearful": (-0.8, 0.7),
    "frantic": (-0.8, 0.9),
    "impatient": (-0.6, 0.7),
    "impulsive": (-0.2, 0.8),
    "intimidating": (-0.7, 0.7),
    "intolerant": (-0.8, 0.5),
    "irritates": (-0.7, 0.6),
    "judgmental": (-0.6, 0.4),
    "over-bearing": (-0.7, 0.6),
    "overwhelming": (-0.4, 0.9),
    "rebellious and obstinate": (-0.6, 0.8),
    "resentful": (-0.8, 0.6),
    "ruthless": (-0.9, 0.7),
    "violent and brutal": (-1.0, 1.0),

    # Low Arousal, Positive Valence (Calm, Relaxed)
    "agreeable": (0.7, -0.3),
    "approachable": (0.7, -0.2),
    "calm": (0.7, -0.6),
    "caring and nurturing": (0.8, -0.3),
    "comforting": (0.9, -0.5),
    "compassion and understanding": (0.9, -0.3),
    "composed": (0.6, -0.5),
    "content": (0.8, -0.6),
    "dependable": (0.6, -0.4),
    "emotional balance and harmony": (0.9, -0.7),
    "empathic": (0.8, -0.2),
    "faithful": (0.7, -0.4),
    "friendly": (0.8, -0.2),
    "genuine": (0.7, -0.4),
    "honest": (0.7, -0.5),
    "kind": (0.9, -0.4),
    "loyal": (0.8, -0.4),
    "peace": (1.0, -0.8),
    "peace of mind": (1.0, -0.9),
    "protective": (0.5, -0.3),
    "pure": (0.8, -0.6),
    "relaxing/removes stress": (0.9, -0.9),
    "reliable and responsible": (0.7, -0.5),
    "safe and non-threatening": (0.9, -0.7),
    "secure": (0.8, -0.6),
    "stable": (0.6, -0.6),
    "supportive": (0.8, -0.3),
    "trustworthy": (0.8, -0.5),
    "warm": (0.8, -0.2),
    "warm-hearted": (0.9, -0.3),

    # Low Arousal, Negative Valence (Sad, Bored)
    "aloof and frigid": (-0.6, -0.4),
    "boring": (-0.5, -0.8),
    "depressed and sad": (-0.9, -0.7),
    "detached": (-0.4, -0.6),
    "disinterested": (-0.4, -0.7),
    "dull": (-0.5, -0.8),
    "empty": (-0.7, -0.8),
    "exhausted": (-0.5, -0.9),
    "indecisive": (-0.3, -0.4),
    "indifferent": (-0.2, -0.6),
    "isolated": (-0.7, -0.5),
    "lifeless": (-0.8, -0.9),
    "lonely": (-0.8, -0.6),
    "melancholy": (-0.7, -0.5),
    "negative": (-0.6, -0.2),
    "pessimistic": (-0.7, -0.3),
    "physically weak": (-0.4, -0.7),
    "tiring": (-0.3, -0.8),
    "unfriendly": (-0.7, -0.3),
    "unimaginative": (-0.5, -0.7),

    # Neutral/Complex/Formal (Center of Map)
    "academic and analytical": (0.2, -0.1),
    "authority": (0.1, 0.4),
    "black": (0.0, 0.0), # Base color
    "classic": (0.4, -0.4),
    "conservative": (0.1, -0.5),
    "conventional": (0.0, -0.6),
    "dignified": (0.5, -0.2),
    "disciplined": (0.2, 0.3),
    "efficient": (0.4, 0.2),
    "elegant": (0.7, -0.1),
    "formal": (0.2, -0.3),
    "hi-tech": (0.3, 0.4),
    "independent": (0.4, 0.1),
    "intelligent": (0.6, 0.1),
    "mature": (0.4, -0.5),
    "mysterious": (-0.1, 0.2),
    "neutral": (0.0, -0.4),
    "practical": (0.3, -0.5),
    "professional": (0.5, -0.1),
    "serious": (-0.1, -0.3),
    "sophisticated": (0.6, -0.1),
    "spiritual": (0.7, -0.3),
    "wealthy": (0.7, 0.2)
}


def _build_color_map_from_data_json(data_path: str = "./data.json") -> dict[str, list[str]]:
    with open(data_path, "r") as f:
        data = json.load(f)

    color_map: dict[str, list[str]] = {}
    for _, details in data.items():
        emotion_text = details.get("text")
        associated_colors = details.get("colors", [])
        for color in associated_colors:
            color = str(color).lower().strip()
            color_map.setdefault(color, []).append(emotion_text)
    return color_map


if __name__ == "__main__":
    # Example: build color->descriptor map + run scent recommender
    try:
        color_map = _build_color_map_from_data_json("./data.json")
        print(color_map)
        engine = ScentRecommender(color_map, va_descriptors)

# Simulate a CNN detecting 80% red and 20% blue
        result = engine.get_scent(color_percentages={"red": 80.0, "blue": 20.0})
        print(f"Final V-A Score: {result['coordinates']}")
        print(f"Recommended Scent: {result['recommendation']}")
    except FileNotFoundError:
        print("data.json not found; skipping scent recommender demo.")