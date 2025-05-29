import json
import os
import math
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('dataset_splitter')


# Function to split the dataset
def split_json(input_file, output_folder, num_parts=6):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the original JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Calculate the size of each part (round up if not perfectly divisible)
    part_size = math.ceil(len(data) / num_parts)

    # Split the data into parts and save each part as a separate JSON file
    for i in range(num_parts):
        start_index = i * part_size
        end_index = min((i + 1) * part_size, len(data))
        part_data = data[start_index:end_index]

        # Define the file name for each part
        part_filename = os.path.join(output_folder, f"part_{i + 1}.json")

        # Write the part data to a JSON file
        with open(part_filename, 'w') as part_file:
            json.dump(part_data, part_file, indent=4)

        logger.info(f"Successfully created dataset part: {part_filename}")


# Example usage
input_file = '../data/train_data.json'  # The path to your large JSON file
output_folder = 'data/split_files'  # The folder to store the split files

split_json(input_file, output_folder)
