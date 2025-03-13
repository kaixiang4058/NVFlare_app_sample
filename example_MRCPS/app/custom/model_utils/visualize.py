import matplotlib.pyplot as plt
import pyvips as vi
import numpy as np

def open_file(org_file_name, tifpage):
    slide = vi.Image.tiffload(org_file_name, page=tifpage)
    region = vi.Region.new(slide)
    data = region.fetch(0, 0, slide.width, slide.height)
    org_file_tiff = np.ndarray(buffer=data, dtype=np.uint8, shape=[
                      slide.height, slide.width, slide.bands]) 
    return org_file_tiff

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(30, 30))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
