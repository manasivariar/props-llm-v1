import os
from collections import Counter

def process_and_clean_datasets(file_path, log_file):
    parameter_counts = Counter()
    unique_params = set()
    cleaned_lines = []
    
    try:
        # Step 1: Read the file and identify duplicates
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if '|' in line:
                params = line.split('|')[0].strip()
                parameter_counts[params] += 1
                
                # If it's the first time seeing this set, add it to cleaned list
                if params not in unique_params:
                    unique_params.add(params)
                    cleaned_lines.append(line)
            else:
                # Keep non-standard lines (headers, blanks)
                cleaned_lines.append(line)
        
        # Step 2: Write duplicates to the verification log
        duplicates = {p: count for p, count in parameter_counts.items() if count > 1}
        if duplicates:
            log_file.write(f"DUPLICATES FOUND IN: {file_path}\n")
            for p, count in duplicates.items():
                log_file.write(f"  - Set: [{p}] appears {count} times\n")
            log_file.write("-" * 50 + "\n")
        
        # Step 3: Remove duplicates by overwriting the file with unique lines
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
            
        return len(duplicates)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

# List of files to check
files = [
    "final_dataset/cliffwalking_dataset.txt", "final_dataset/hopper_dataset.txt", 
    "final_dataset/inverted_double_pendulum_dataset.txt", "final_dataset/mountaincarcontinuous_dataset.txt",
    "final_dataset/nav_dataset.txt", "final_dataset/pong_dataset.txt", 
    "final_dataset/reacher_dataset.txt", "final_dataset/swimmer_dataset.txt", "final_dataset/walker2d_dataset.txt"
]

# Open the verification file and process each dataset
with open("duplicates_for_verification.txt", "w") as verify_log:
    verify_log.write("=== DUPLICATE PARAMETER VERIFICATION LOG ===\n\n")
    
    for file_name in files:
        if os.path.exists(file_name):
            print(f"Checking and cleaning {file_name}...")
            num_dupes = process_and_clean_datasets(file_name, verify_log)
            if num_dupes > 0:
                print(f"  -> Found {num_dupes} unique parameter sets with duplicates. Logged for verification.")
            else:
                print("  -> No duplicates found.")
        else:
            print(f"File not found: {file_name}")

print("\nTask Complete.")
print("1. Please check 'duplicates_for_verification.txt' to verify the duplicates.")
print("2. The original dataset files have now been cleaned (only unique entries remain).")