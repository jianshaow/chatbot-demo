import base64
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image

from common.prompts import mm_image_url


def show_demo_image():
    print("-" * 80)
    response = requests.get(mm_image_url, timeout=2)
    print("demo image URL:", mm_image_url)

    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()

    base64_content = base64.b64encode(response.content).decode("utf-8")
    return response.content, base64_content


if __name__ == "__main__":
    show_demo_image()
