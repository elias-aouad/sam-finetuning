import os
import time
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By


def scroll_to_load_more_images(driver):
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)


def scrapping(url: str, output_folder: str, max_images: int):
    os.makedirs(output_folder, exist_ok=True)

    driver = webdriver.Chrome()
    driver.get(url)

    scroll_to_load_more_images(driver=driver)

    img_elements = driver.find_elements(By.CLASS_NAME, 'rg_i')

    for i, img_element in enumerate(img_elements):
        img_url = img_element.get_attribute("src")
        img_name = f"{i}.jpg"
        img_path = os.path.join(output_folder, img_name)

        if img_url and img_url.startswith("http"):
            response = requests.get(img_url, stream=True)

            with open(img_path, "wb") as img_file:
                for chunk in response.iter_content(chunk_size=128):
                    img_file.write(chunk)

            print(f"Downloaded: {img_name}")
        
        if i > max_images:
            break

    driver.quit()


def main():
    # This url should be copied from a google search on images
    # The query was "packshot images white background"
    # PS : It could have also been a Duckduckgo search
    url = ""
    output_folder = ""

    scrapping(url=url, output_folder=output_folder, max_images=1000)


if __name__ == "__main__":
    main()
