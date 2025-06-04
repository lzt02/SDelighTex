import os
import sys
import importlib
import click

# Add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@click.command()
@click.option('--source_dir', '-s', 
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              required=True,
              help='Source directory containing images and mesh')
@click.option('--method', '-m',
              type=click.Choice(['metashape'], case_sensitive=False),
              default='metashape',
              show_default=True,
              help='Texturing method to use')
@click.option('--texture_size',
              type=click.IntRange(1024, 8192),
              default=4096,
              show_default=True,
              help='Texture map resolution')
def main(source_dir, method, texture_size):
    """Texture mesh using different methods"""
    # Path handling
    images_folder = os.path.join(source_dir, "images_delighted")
    cameras_folder = os.path.join(source_dir, "sparse", "0")
    
    # Check if paths exist
    if not all(os.path.exists(p) for p in [images_folder, cameras_folder]):
        images_folder = os.path.join(source_dir + "_source_data", "images_delighted")
        cameras_folder = os.path.join(source_dir + "_source_data", "sparse", "0")
        
    if not os.path.exists(images_folder) or not os.path.exists(cameras_folder):
        print(f"Error: Images folder or cameras folder not found")
        print(f"Checked paths: {images_folder} and {cameras_folder}")
        return
    
    obj_file = os.path.join(source_dir, "3DModel.obj")
    if not os.path.exists(obj_file):
        print(f"Error: Mesh file not found at {obj_file}")
        return

    # Generate output filename
    folder_name = os.path.basename(images_folder)
    filename = f"{folder_name.split('_', 1)[-1]}.obj"
    output_file = os.path.join(source_dir, filename)

    # Check if output exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Exiting.")
        return

    # Dynamically load method module
    try:
        module_name = f"system.{method}"
        print(f"Loading module: {module_name}")
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Could not import module for method {method}: {e}")
        return

    # Initialize processor
    print(f"Initializing {method} texturing processor...")
    processor = module.TextureProcessor(enable_gpu=True) 
    
    # Run texturing
    print(f"Starting mesh texturing with {texture_size}px texture...")
    processor.texture_mesh(
        images_folder, 
        cameras_folder, 
        obj_file, 
        output_file,
        texture_size=texture_size
    )
    
    # Verify result
    if os.path.exists(output_file):
        print(f"Successfully created textured model at {output_file}")
    else:
        print(f"Error: Output file not created {output_file}")

if __name__ == "__main__":
    main()