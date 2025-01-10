from PIL import Image
import os
import sys

def convert_bmp_to_png(input_folder, output_folder):
    """
    Convert all BMP images in the input folder to PNG format and save them in the output folder.

    Args:
        input_folder (str): Path to the folder containing BMP images.
        output_folder (str): Path to the folder to save PNG images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            # Construct full file path
            bmp_path = os.path.join(input_folder, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_folder, png_filename)

            try:
                # Open the BMP image and convert to PNG
                with Image.open(bmp_path) as img:
                    img.save(png_path, 'PNG')
                print(f"Converted: {filename} -> {png_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    # Define the input and output folder paths
    input_folder = os.path.dirname(os.path.abspath(__file__))  # Current folder
    output_folder = input_folder  # Same folder for output

    convert_bmp_to_png(input_folder, output_folder)
