from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image

from common import demo_image_url

DPI = (72, 72)


def show_demo_image():
    print("-" * 80)
    response = requests.get(demo_image_url, timeout=5)
    print("demo image URL:", demo_image_url)

    img = Image.open(BytesIO(response.content))
    show_image(img)


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
