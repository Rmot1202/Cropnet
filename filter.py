import os
import re

# Define root directory
root_dir = '/Users/RavenMott1/Downloads/Cropnet/data/New Folder With Items/'

# Hold matched files with state and month info
matched_files = []

# Walk through all files in the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.csv') and '2022' in filename:
            full_path = os.path.join(dirpath, filename)
            
            # Extract state and month
            try:
                parts = full_path.split(os.sep)
                state = parts[-2]
                match = re.search(r'2022-(\d{2})', filename)
                month = match.group(1) if match else '00'
                matched_files.append((state, month, full_path))
            except Exception as e:
                print(f"Skipping {full_path}: {e}")

# Sort by state and then month
matched_files.sort(key=lambda x: (x[0], x[1]))

# Format the list
formatted = ', '.join(f'"{path}"' for _, _, path in matched_files)

# Save to a text file
output_path = 'filtered_2022_files.txt'
with open(output_path, 'w') as f:
    f.write(formatted)

print(f"Saved to {output_path}")
