import os
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from download_maps import load_config_file

def infer_resource_id(map_id):
    return map_id.rstrip("abcdefghijklmnopqrstuvwxyz")

def convert_pdf_pages_to_image(pdf_path, page_numbers, rotation=0, dpi=300):
    # Convert only the desired pages (1-indexed for pdf2image)
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_numbers[0], last_page=page_numbers[-1])
    image = images[0]

    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    return image

def remove_empty_columns(image, threshold=250):
    """
    Removes all columns in the image where all pixels are above a brightness threshold.
    Some images may contain white space to account for the crease in a pamphlet.
    """
    img_np = np.array(image.convert("RGB"))
    gray = np.mean(img_np, axis=2)
    non_empty_cols = np.any(gray < threshold, axis=0)

    if np.count_nonzero(non_empty_cols) == 0:
        raise ValueError("All columns are empty. Check your input or threshold.")

    cleaned_np = img_np[:, non_empty_cols]
    return Image.fromarray(cleaned_np)


def main(manifest_path, pdf_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = load_config_file(manifest_path)

    for entry in manifest:
        map_id = entry['id']
        resource_id = infer_resource_id(map_id)
        pdf_file = os.path.join(pdf_dir, f"{resource_id}.pdf")
        output_file = os.path.join(output_dir, f"{map_id}_raw.png")
        if os.path.exists(output_file):
            print(f"{output_file} already exists, skipping...")
        else:
            print(f"[→] Converting pages {entry['pages']} of {entry['id']}...")
            img = convert_pdf_pages_to_image(pdf_file, entry["pages"], rotation=entry.get("rotation", 0))
            if entry.get("remove_empty_columns", False):
                print(f"[🧽] Removing whitespace columns for {entry['id']}")
                img = remove_empty_columns(img, threshold=200)
            img.save(output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='maps_manifest.yaml')
    parser.add_argument('--pdf_dir', type=str, default='originals')
    parser.add_argument('--output_dir', type=str, default='images_raw')
    args = parser.parse_args()

    main(args.manifest, args.pdf_dir, args.output_dir)
