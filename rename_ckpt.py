#!/usr/bin/env python3


# --- path configuration (see .env.example) ---
import os
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/path/to/checkpoints")

import os
import sys
from pathlib import Path
import torch
import shutil
import argparse

def rename_checkpoint_folders(base_dir="/xx", new_head="checkpoint-977"):
    """
    Rename all subfolders under base_dir that start with 'checkpoint' to 'lcheckpoint'
    """
    base_path = Path(base_dir)
    
    # Check if base directory exists
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    if not base_path.is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)
    
    # Find all directories starting with "checkpoint"
    checkpoint_dirs = list(base_path.rglob("checkpoint*"))
    checkpoint_dirs = [d for d in checkpoint_dirs if d.is_dir()]
    renamed_count = 0
    skipped_count = 0
    if not checkpoint_dirs:
        print("No directories starting with 'checkpoint' found")
    
    else:
        for dir_path in checkpoint_dirs:
            # Create new name by prepending 'l'
            if dir_path.name == new_head:
                print(f"{dir_path} is the new head")
                skipped_count += 1
                continue

            new_name = f"l{dir_path.name}"
            new_path = dir_path.parent / new_name
            
            # Check if target doesn't already exist
            if not new_path.exists():
                try:
                    print(f"Renaming: {dir_path} -> {new_path}")
                    dir_path.rename(new_path)
                    print(f"Success: Renamed {dir_path.name} to {new_name}")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error: Failed to rename {dir_path}: {e}")
            else:
                print(f"Warning: {new_path} already exists, skipping {dir_path}")
                skipped_count += 1
        

    # Find if any subfolder has the name l{new_head}, and change it to new_head
    lnew_head_dir = base_path / f"l{new_head}"
    new_head_dir = base_path / new_head

    if lnew_head_dir.exists() and lnew_head_dir.is_dir():
        if not new_head_dir.exists():
            try:
                print(f"Renaming: {lnew_head_dir} -> {new_head_dir}")
                lnew_head_dir.rename(new_head_dir)
                print(f"Success: Renamed {lnew_head_dir.name} to {new_head}")
                renamed_count += 1
            except OSError as e:
                print(f"Error: Failed to rename {lnew_head_dir}: {e}")
        else:
            print(f"Warning: {new_head_dir} already exists, skipping {lnew_head_dir}")
            skipped_count += 1

    print(f"\nRename operation completed.")
    print(f"Renamed: {renamed_count} folders")
    print(f"Skipped: {skipped_count} folders")

if __name__ == "__main__":
    # You can change the base directory here if needed
    parser = argparse.ArgumentParser(description="Rename checkpoint folders and move files.")
    parser.add_argument("--base", type=str, help="Base directory to operate on")
    parser.add_argument("--head", type=str, default=None, help="New head checkpoint folder name")
    parser.add_argument("--target", type=str, default=None, help="Subfolder to move files into")

    args = parser.parse_args()

    base_directory = args.base
    # base_directory = CKPT_ROOT + "/qwen3_0_6b_mixpretrain_stage1/"
    # new_head = "checkpoint-9770"
    if args.head != None:
        rename_checkpoint_folders(base_directory, args.head)

    # Move all files under base_directory into a new subfolder named 'acheckpoint-9770'
    if args.target != None:
        target_subfolder = Path(base_directory) / args.target
        # target_subfolder = Path(base_directory) / "acheckpoint-9770"
        target_subfolder.mkdir(exist_ok=True)

        for item in Path(base_directory).iterdir():
            if item.is_file():
                shutil.move(str(item), str(target_subfolder / item.name))

        print(f"All files moved to {target_subfolder}")

# python rename_ckpt.py --base CKPT_ROOT + "/qwen3_0.6b_mixpretrain_stage1_1_1_0.8/" --head "checkpoint-7824" --target "acheckpoint-7824"