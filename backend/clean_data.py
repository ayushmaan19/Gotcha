import pandas as pd
import glob
import os

def clean_and_prepare_data():
    # 1. Find all session CSVs in the current folder (dataset_logs)
    csv_files = glob.glob("dataset_logs/session_*.csv")
    
    print(f"[*] Found {len(csv_files)} raw CSV files.")
    
    df_list = []
    
    # 2. Load and filter broken/empty files
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Skip files with basically no data (like session_20260219_034932.csv)
            if len(df) > 10:
                df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    if not df_list:
        print("[!] No valid data found.")
        return

    # 3. Combine into one master dataframe
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"[*] Combined raw data shape: {master_df.shape}")
    
    # 4. Handle Missing Values (Drop rows where crucial biometrics failed to record)
    master_df = master_df.dropna()
    
    # 5. REMOVE DATA LEAKAGE & NOISE COLUMNS
    # Dropping timestamp (noise) and final_integrity_score (leakage)
    columns_to_drop = ['timestamp', 'final_integrity_score']
    master_df = master_df.drop(columns=[col for col in columns_to_drop if col in master_df.columns])
    
    # 6. Ensure label is integer
    master_df['label'] = master_df['label'].astype(int)
    
    # 7. Shuffle the data thoroughly
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 8. Save the perfectly clean, ML-ready dataset
    output_name = "cleaned_training_data.csv"
    master_df.to_csv(output_name, index=False)
    
    print(f"\n[✓] SUCCESS! Cleaned dataset saved as: {output_name}")
    print(f"[*] Final Shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
    print("[*] Class Balance:")
    print(master_df['label'].value_counts())

if __name__ == "__main__":
    clean_and_prepare_data()