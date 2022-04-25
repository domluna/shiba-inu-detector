from io import BytesIO
from pathlib import Path

import fire
import requests
from PIL import Image, UnidentifiedImageError

SEARCH_URL = "https://huggingface.co/api/experimental/images/search"


def get_image_urls_by_term(search_term: str, count=150):
    params = {
        "q": search_term,
        "license": "public",
        "imageType": "photo",
        "count": count,
    }
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    response_data = response.json()
    image_urls = [img["thumbnailUrl"] for img in response_data["value"]]
    return image_urls


# TODO: make this async
def gen_images_from_urls(urls):
    num_skipped = 0
    for url in urls:
        response = requests.get(url)
        if not response.status_code == 200:
            num_skipped += 1
        try:
            img = Image.open(BytesIO(response.content))
            yield img
        except UnidentifiedImageError:
            num_skipped += 1

    print(f"Retrieved {len(urls) - num_skipped} images. Skipped {num_skipped}.")


def urls_to_image_folder(urls, save_directory):
    for i, image in enumerate(gen_images_from_urls(urls)):
        image.save(save_directory / f"{i}.jpg")


def generate_data(data_dir, search_terms_file, count=150):
    images_dir = Path(data_dir) / Path("images")

    with open(search_terms_file, "r") as f:
        search_terms = f.read().splitlines()

    for search_term in search_terms:
        search_term = search_term.strip()
        search_term_dir = images_dir / search_term
        urls = get_image_urls_by_term(search_term, count=count)
        if len(urls) == 0:
            print(f"No images found for {search_term}")
            continue
        search_term_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving images of {search_term} to {str(search_term_dir)}...")
        urls_to_image_folder(urls, search_term_dir)


if __name__ == "__main__":
    fire.Fire(generate_data)
