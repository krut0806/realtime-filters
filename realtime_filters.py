import cv2
import numpy as np

# === Basic Filters ===

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_canny(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_sepia(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img_bgr, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB)

def apply_invert(image):
    return cv2.bitwise_not(image)

def apply_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def apply_cartoon(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img_bgr, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

# === 2000s-Inspired Filters ===

def apply_blue_teal(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    b = cv2.addWeighted(b, 1.2, b, 0, 0)
    r = cv2.addWeighted(r, 0.9, r, 0, 0)
    result = cv2.merge([b, g, r])
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def apply_matrix_green(image):
    green = np.zeros_like(image)
    green[:, :, 1] = image[:, :, 1]  # Only keep green channel
    return green

def apply_vintage_film(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sepia = np.array([[0.393, 0.769, 0.189],
                      [0.349, 0.686, 0.168],
                      [0.272, 0.534, 0.131]])
    filtered = cv2.transform(img_bgr, sepia)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    filtered = cv2.GaussianBlur(filtered, (3, 3), 0)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

# === Soft Glow Filter ===

def apply_soft_glow(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_float = img_bgr.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img_float, (21, 21), 0)
    glow = cv2.addWeighted(img_float, 0.7, blur, 0.6, 0)
    glow = np.clip(glow * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(glow, cv2.COLOR_BGR2RGB)
