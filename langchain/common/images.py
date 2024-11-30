import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def show_demo_image():
    image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

    img_response = requests.get(image_url)
    print("image URL:", image_url)

    img = Image.open(BytesIO(img_response.content))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    show_demo_image()
