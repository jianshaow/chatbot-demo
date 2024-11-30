import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from common import demo_image_url


def show_demo_image():
    print("-" * 80)
    response = requests.get(demo_image_url)
    print("demo image URL:", demo_image_url)

    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    show_demo_image()
