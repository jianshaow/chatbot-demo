import base64
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image

from common.prompts import mm_image_url


def show_demo_image():
    print("-" * 80)
    print("demo image URL:", mm_image_url)
    response = requests.get(mm_image_url, timeout=2)
    assert response.status_code == 200
    bytes_content = response.content

    img = Image.open(BytesIO(bytes_content))
    plt.imshow(img)
    plt.show()

    base64_content = base64.b64encode(bytes_content).decode("utf-8")
    return bytes_content, base64_content


if __name__ == "__main__":
    show_demo_image()
