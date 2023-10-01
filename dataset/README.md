# SAM Dataset

Hi, this is a repository for scrapping images and annotating these images with a background.

The operations are to be exececuted in this order :


1. Scrap images :

    - Make a Google (or Duckduckgo) search about packshot images

    - Copy the url in the `scrap_images.py` file

    - Define an output folder in the `scrap_images.py` file

    - Execute the file


2. Apply transparent background

   You can use [this](https://github.com/plemeri/transparent-background) repository for removing the background from images

```commandline
transparent-background --source <source-folder> --dest <destination-folder> --mode base
```

3. Get masks from RGBA images (`get_masks.py` file)

    - Copy the directory of the RGBA images and apply it to the `dir_rgba_images` variable

    - Select the directories for saving masks as arrays (`dir_mask_array`) and saving masks as images (`dir_mask_image`)

    - Execute the file
