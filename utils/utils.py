import os
import csv
from typing import List, Tuple, Union, Dict
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


def setup_folders_substrate(output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f"{output_dir}/data_analysis"):
        os.mkdir(f"{output_dir}/data_analysis")
    if not os.path.exists(f"{output_dir}/coordinates"):
        os.mkdir(f"{output_dir}/coordinates")

def setup_folders_evolution(output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f"{output_dir}/video"):
        os.mkdir(f"{output_dir}/video")
    if not os.path.exists(f"{output_dir}/topology"):
        os.mkdir(f"{output_dir}/topology")



def save_data_sources(data_sources: Dict[str, np.ndarray], filepath: str):
    try:
        # The **data_sources syntax unpacks the dictionary into keyword arguments,
        # which is exactly what np.savez_compressed expects.
        # e.g., np.savez_compressed(filepath, trained=array1, random=array2)
        np.savez_compressed(filepath, **data_sources)
        print(f"Successfully saved data sources to: {filepath}")

    except IOError as e:
        print(f"Error: Could not write to file at {filepath}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving: {e}")


def load_data_sources(filepath: str) -> Dict[str, np.ndarray]:
    try:
        # np.load returns an NpzFile object, which acts like a dictionary
        loaded_data = np.load(filepath)
        
        # Reconstruct a standard Python dictionary from the NpzFile object
        data_sources = {key: loaded_data[key] for key in loaded_data.files}
        
        print(f"Successfully loaded data sources from: {filepath}")
        return data_sources

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Returning an empty dictionary.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading: {e}")
        return {}

FIELDNAMES = ["dimensionality", "sampling", "method", "max_fitness"]

def append_summary_row(path: str, row: dict, fieldnames=FIELDNAMES) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
