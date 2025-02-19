from matplotlib import pyplot as plt
from PIL import Image

DPI = (72, 72)


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
