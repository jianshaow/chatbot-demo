import base64
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image

from common.prompts import mm_image_url

DPI = (72, 72)


def show_demo_image():
    print("-" * 80)
    print("demo image URL:", mm_image_url)
    response = requests.get(mm_image_url, timeout=2)
    assert response.status_code == 200
    bytes_content = response.content

    img = Image.open(BytesIO(bytes_content))
    show_image(img)

    base64_content = base64.b64encode(bytes_content).decode("utf-8")
    return bytes_content, base64_content


def show_image(image: Image):
    print("image size:", image.size)
    dpi = image.info.get("dpi", DPI)
    width, height = image.size

    fig_width = width / dpi[0]
    fig_height = height / dpi[1]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi[0])
    fig.gca().set_position([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    plt.show()


if __name__ == "__main__":
    show_demo_image()
