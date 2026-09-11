import os
import shutil
from pathlib import Path
import argparse

def move_jsonl_files(main_folder):
    """
    Find all .jsonl files in subfolders and move them to the main folder
    with renamed format: {subfoldername}_{originalname}
    
    Args:
        main_folder: Path to the main folder containing subfolders
    """
    main_folder = Path(main_folder)
    
    # Check if the main folder exists
    if not main_folder.exists():
        print(f"Error: The folder '{main_folder}' does not exist.")
        return
    
    moved_count = 0
    subfolders_to_delete = []
    # Iterate through all items in the main folder
    for item in main_folder.iterdir():
        # Check if it's a directory (subfolder)
        if item.is_dir():
            subfolder_name = item.name
            subfolders_to_delete.append(item)
            
            # Find all .jsonl files in this subfolder (including nested subfolders)
            for jsonl_file in item.rglob('*.jsonl'):
                # Get the original filename
                original_name = jsonl_file.name
                
                # Create new filename with subfolder prefix
                new_name = f"{subfolder_name}_{original_name}"
                new_path = main_folder / new_name
                
                # Handle duplicate filenames
                counter = 1
                while new_path.exists():
                    name_without_ext = original_name.rsplit('.jsonl', 1)[0]
                    new_name = f"{subfolder_name}_{name_without_ext}_{counter}.jsonl"
                    new_path = main_folder / new_name
                    counter += 1
                
                # Move the file
                try:
                    shutil.move(str(jsonl_file), str(new_path))
                    print(f"Moved: {jsonl_file.relative_to(main_folder)} → {new_name}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {jsonl_file}: {e}")
    
    print(f"\nTotal files moved: {moved_count}")

    # Delete the subfolders
    print("\nDeleting subfolders...")
    deleted_count = 0
    for subfolder in subfolders_to_delete:
        try:
            shutil.rmtree(subfolder)
            print(f"Deleted: {subfolder.name}/")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {subfolder.name}: {e}")
    
    print(f"\nTotal subfolders deleted: {deleted_count}")

# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    args = argparse.ArgumentParser()
    args.add_argument("--folder_path", type=str, required=True, help="Path to the main folder")
    args = args.parse_args()
    folder_path = args.folder_path
    move_jsonl_files(folder_path)

# move everything to one folder first: 
# mkdir /path/to/data/laion_tokenized_gigatok
# mv /path/to/data/laion_part_{1..8}_tokenized_gigatok /path/to/data/laion_tokenized_gigatok/
# python laion_movejsonl.py --folder_path /path/to/data/laion_tokenized_gigatok