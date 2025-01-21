#!/bin/bash

# Function to remove broken Git objects
remove_broken_objects() {
    echo "Checking for broken objects..."
    
    # Run git fsck to identify broken objects
    git fsck --full 2>&1 | grep -E 'error: object [a-f0-9]{40}' | while read -r line; do
        # Extract the object hash from the error message
        object_hash=$(echo "$line" | grep -oE '[a-f0-9]{40}')
        
        if [ -n "$object_hash" ]; then
            # Calculate the directory and file path for the object
            object_dir=${object_hash:0:2}
            object_file=${object_hash:2}
            object_path=".git/objects/$object_dir/$object_file"
            
            if [ -f "$object_path" ]; then
                echo "Removing broken object: $object_path"
                rm -f "$object_path"
            else
                echo "Object $object_hash not found in .git/objects directory. Skipping."
            fi
        fi
    done
}

# Main script
main() {
    echo "Starting cleanup of broken Git objects..."
    remove_broken_objects
    
    echo "Performing garbage collection to clean up dangling references..."
    git gc --prune=now --aggressive

    echo "Cleanup complete. Run 'git fsck --full' again to verify."
}

# Check if the script is being run inside a Git repository
if [ ! -d ".git" ]; then
    echo "Error: This script must be run inside a Git repository."
    exit 1
fi

# Execute the main function
main

