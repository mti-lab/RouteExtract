import os
from PIL import Image
from download_maps import load_config_file


def crop_and_resize_image(input_path, crop_box, final_size):
    image = Image.open(input_path)

    # Apply crop: [left, upper, right, lower]
    cropped = image.crop(crop_box)

    # Compute new size that preserves aspect ratio
    target_w, target_h = final_size
    orig_w, orig_h = cropped.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize while preserving aspect ratio
    resized = cropped.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create white background canvas
    canvas = Image.new("RGB", final_size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas




def main(manifest_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = load_config_file(manifest_path)

    for entry in manifest:
        input_file = os.path.join(input_dir, f"{entry['id']}_raw.png")
        output_file = os.path.join(output_dir, f"{entry['id']}.png")

        crop_box = entry["crop_box"]
        final_size = tuple(entry.get("final_size", [1024, 1024]))

        print(f"[✂️ ] Cropping and resizing {entry['id']}...")
        img = crop_and_resize_image(input_file, crop_box, final_size)
        img.save(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='maps_manifest.yaml')
    parser.add_argument('--input_dir', type=str, default='images_raw')
    parser.add_argument('--output_dir', type=str, default='images_final')
    args = parser.parse_args()

    main(args.manifest, args.input_dir, args.output_dir)
