import gradio as gr
import cv2
import numpy as np


def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_canny(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia

def apply_invert(image):
    return cv2.bitwise_not(image)

def apply_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def apply_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_blue_teal(image):
    b, g, r = cv2.split(image)
    b = cv2.addWeighted(b, 1.2, b, 0, 0)
    r = cv2.addWeighted(r, 0.9, r, 0, 0)
    result = cv2.merge([b, g, r])
    return result

def apply_matrix_green(image):
    green = np.zeros_like(image)
    green[:, :, 1] = image[:, :, 1]
    return green

def apply_vintage_film(image):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    filtered = cv2.transform(image, kernel)
    filtered = cv2.GaussianBlur(filtered, (3, 3), 0)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def apply_soft_glow(image):
    img_float = image.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img_float, (21, 21), 0)
    glow = cv2.addWeighted(img_float, 0.7, blur, 0.6, 0)
    glow = np.clip(glow * 255, 0, 255).astype(np.uint8)
    return glow


def apply_filter(image, filter_name):
    if image is None:
        return None
    filters = {
        "Grayscale": apply_grayscale,
        "Canny Edge": apply_canny,
        "Sepia": apply_sepia,
        "Invert": apply_invert,
        "Pencil Sketch": apply_sketch,
        "Cartoon": apply_cartoon,
        "Blue Teal": apply_blue_teal,
        "Matrix Green": apply_matrix_green,
        "Vintage Film": apply_vintage_film,
        "Soft Glow": apply_soft_glow
    }
    return filters.get(filter_name, lambda x: x)(image)


filter_options = [
    "None",
    "Grayscale",
    "Canny Edge",
    "Sepia",
    "Invert",
    "Pencil Sketch",
    "Cartoon",
    "Blue Teal",
    "Matrix Green",
    "Vintage Film",
    "Soft Glow"
]

with gr.Blocks() as demo:
    gr.Markdown("##  Real-Time Webcam Filters (2000s Aesthetic)")

    with gr.Row():
        webcam = gr.Image(label="Webcam Snapshot", type="numpy")
        filter_choice = gr.Dropdown(choices=filter_options, value="None", label="🎛️ Choose a Filter")

    output = gr.Image(label=" Filtered Output")

    apply_btn = gr.Button(" Apply Filter")

    def run_filter(img, choice):
        if choice == "None":
            return img
        return apply_filter(img, choice)

    apply_btn.click(fn=run_filter, inputs=[webcam, filter_choice], outputs=output)

demo.launch()
