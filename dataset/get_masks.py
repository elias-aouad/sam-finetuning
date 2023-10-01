import os
import cv2
import numpy as np
from tqdm import trange
from PIL import Image


def save_masks(
    dir_rgba_images, dir_mask_array, dir_mask_image, threshold=5
):
    filenames = os.listdir(dir_rgba_images)
    num_images = len(filenames)

    for i in trange(num_images):

        rgba_path = os.path.join(dir_rgba_images, f"{i}_rgba.png")
        rgba_array = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)

        mask = (rgba_array[:, :, -1] >= threshold).astype("int")
        mask_image = (255 * mask).astype("uint8")

        # save info
        path_array = os.path.join(dir_mask_array, f"{i}.npy")
        path_image = os.path.join(dir_mask_image, f"{i}.jpg")

        with open(path_array, 'wb') as f:
            np.save(f, mask)

        Image.fromarray(mask_image).save(path_image, format='JPEG', quality=95)


def main():
    # Directory of RGBA images (processed by transparent background)
    dir_rgba_images = ""

    # Directory in which we save masks as arrays
    dir_mask_array = ""

    # Directory in which we save masks as JPEG images
    dir_mask_image = ""

    os.makedirs(dir_mask_array, exist_ok=True)
    os.makedirs(dir_mask_image, exist_ok=True)

    save_masks(
        dir_rgba_images=dir_rgba_images, 
        dir_mask_array=dir_mask_array, 
        dir_mask_image=dir_mask_image
    )


if __name__ == "__main__":
    main()
