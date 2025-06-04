import os
import argparse
import torch
from PIL import Image
from sdelightex.utils.images import align_saturation
import click

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
def main(source_dir):

    input_folder = rf"{source_dir}/images"
    output_folder = rf"{source_dir}/images_delighted"

    # Call the function with the provided arguments
    delight_images(input_folder, output_folder)
    align_saturation(input_folder, output_folder, output_folder, group_size=5, ext=("png", "jpg", "jpeg"))


if __name__ == "__main__":
    main()
