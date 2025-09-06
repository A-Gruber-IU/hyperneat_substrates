import os
import csv
from typing import List, Tuple, Union
import numpy as np

def save_coordinates_to_csv(
    coordinates: Union[List[Tuple], np.ndarray], 
    filepath: str
):
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            num_dims = len(coordinates[0])
            header = [f"dim_{i+1}" for i in range(num_dims)]
            writer.writerow(header)
            
            # Write all the coordinate rows
            writer.writerows(coordinates)
            
        print(f"Successfully saved coordinates to: {filepath}")

    except IOError as e:
        print(f"Error: Could not write to file at {filepath}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def setup_folder_structure(output_dir: str):
    if not os.path.exists(f"output"):
        os.mkdir("output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f"{output_dir}/data_analysis"):
        os.mkdir(f"{output_dir}/data_analysis")
    if not os.path.exists(f"{output_dir}/coordinates"):
        os.mkdir(f"{output_dir}/coordinates")
    if not os.path.exists(f"{output_dir}/video"):
        os.mkdir(f"{output_dir}/video")
    if not os.path.exists(f"{output_dir}/topology"):
        os.mkdir(f"{output_dir}/topology")