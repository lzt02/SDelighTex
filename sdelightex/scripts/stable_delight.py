import os
import argparse
import torch
from PIL import Image
from sdelightex.utils.images import align_saturation, detail_transfer, create_brightness_mask
import click
import numpy as np
import time


def delight_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the predictor
    predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path):
                #print(f"Skipping {filename} as it already exists")
                continue

            # Load the image
            input_image = Image.open(input_path)

            # Apply the model to the image
            delight_image = predictor(input_image)

            # Save the result
            delight_image.save(output_path)
            #print(f"Processed and saved {filename} to {output_path}")


@click.command()
@click.option('--source_dir', '-s', 
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              required=True,
              help='Input directory containing images')
@click.option('--mode', '-m',
              type=click.Choice(['delight', 'detail']),
              default='delight',
              help='Processing mode: delight (prioritize highlight removal but may lose details) or detail (prioritize detail preservation but may retain some highlights)')
def main(source_dir, mode):
    start_time = time.time()
    input_folder = rf"{source_dir}/images"
    output_folder = rf"{source_dir}/images_delighted_detail"

    # Call the function with the provided arguments
    delight_images(input_folder, output_folder)
    

    align_saturation(output_folder, input_folder, output_folder, group_size=5, ext=("png", "jpg", "jpeg"))
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    input_files = [f for f in os.listdir(input_folder) if f.endswith(exts)]
    for fname in input_files:
        input_path = os.path.join(input_folder, fname)
        target_path = os.path.join(output_folder, fname)
        img_input = Image.open(input_path).convert('RGB')
        img_output = Image.open(target_path).convert('RGB')
    
        if mode == 'delight':
            mask = create_brightness_mask(input_path)
        elif mode == 'detail':
            mask = np.ones(img_input.size[::-1], dtype=np.float32)
        
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        img_input_rgba = img_input.copy()
        img_input_rgba.putalpha(mask_img)
        result = detail_transfer(
            source_image=img_input_rgba,
            target_image=img_output,
            mode="add",
            blur_sigma=1,
            blend_factor=1,
            mask=None,
            use_alpha_as_mask=True
        )
        save_path = os.path.join(output_folder, fname)
        result.save(save_path)
    end_time = time.time()
    print(f"Delight Time Taken: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
