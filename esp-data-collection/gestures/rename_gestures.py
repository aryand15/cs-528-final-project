import os
import re

# Your three folders
folders = ["cameron"]

offset = 34

pattern = re.compile(r"^(.*?)_(\d+)\.txt$")

for folder in folders:
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        
        if match:
            gesture = match.group(1)
            number = int(match.group(2))

            new_number = number + offset

            # Preserve leading zeros (2 digits here)
            new_filename = f"{gesture}_{new_number:02d}.txt"

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)

            os.rename(old_path, new_path)
            print(f"{filename} -> {new_filename}")