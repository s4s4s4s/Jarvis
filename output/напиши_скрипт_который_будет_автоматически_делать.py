import shutil
import os

def backup_files(src_dir, dest_dir):
    """
    Backup all files in the source directory to the destination directory.
    
    :param src_dir: Source directory containing files to be backed up.
    :param dest_dir: Destination directory where backups will be stored.
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    # Walk through all directories and files in the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
            
            # Ensure the destination path's directory exists
            dest_directory = os.path.dirname(dest_file_path)
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)
                
            shutil.copy2(src_file_path, dest_file_path)  # copy2 preserves metadata

# Example usage:
backup_files('/path/to/source', '/path/to/destination')
