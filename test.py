import os
import pandas as pd
import shutil

def process_file(**kwargs):
    raw_data_path = kwargs.get('raw_data_path', '/opt/data/raw-data/')
    good_data_path = kwargs.get('good_data_path', '/opt/data/good-data/')
    bad_data_path = kwargs.get('bad_data_path', '/opt/data/bad-data/')

    os.makedirs(good_data_path, exist_ok=True)
    os.makedirs(bad_data_path, exist_ok=True)

    
    files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(raw_data_path, file)
        try:
            # Implement your data quality checks here. This is a simple check for non-empty files.
            df = pd.read_csv(file_path)
            if not df.empty:
                shutil.move(file_path, os.path.join(good_data_path, file))
            else:
                shutil.move(file_path, os.path.join(bad_data_path, file))
        except Exception as e:
            # Move files that cause errors to bad_data_path
            shutil.move(file_path, os.path.join(bad_data_path, file))
            print(f"Error processing file {file}: {e}")

process_file(raw_data_path= '../../data/raw-data',
             good_data_path = '../../data/good-data',
             bad_data_path = '../../data/bad-data'
             )