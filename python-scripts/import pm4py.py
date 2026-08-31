import gzip
import shutil
import os

input_folder = r"C:\Users\Lenovo\Desktop\UoL\SEM-2\DataMining\CW2\\note_gz_csv"
output_folder = r"C:\Users\Lenovo\Desktop\UoL\SEM-2\DataMining\CW2\\note"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv.gz"):
        gz_path = os.path.join(input_folder, file)
        csv_path = os.path.join(output_folder, file[:-3])  # remove .gz
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Extracted: {file}")

print("All files extracted.")