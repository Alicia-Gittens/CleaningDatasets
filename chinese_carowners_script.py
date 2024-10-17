import pandas as pd
import re
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Translation of foreign language headers to English
translated_headers = {
    '车架号': 'VIN',
    '姓名': 'Name',
    '身份证': 'ID Number',
    '性别': 'Gender',
    '手机': 'Mobile Phone',
    '邮箱': 'Email',
    '省': 'Province',
    '城市': 'City',
    '地址': 'Address',
    '邮编': 'Postal Code',
    '生日': 'Date of Birth',
    '行业': 'Industry',
    '月薪': 'Monthly Salary',
    '婚姻': 'Marital Status',
    '教育': 'Education',
    'BRAND': 'Brand',
    '车系': 'Car Series',
    '车型': 'Car Model',
    '配置': 'Configuration',
    '颜色': 'Color',
    '发动机号': 'Engine Number'
}

# Configuration
columns_to_remove_in_clean = ['gender', 'industry', 'monthly_salary', 'marital_status', 'education', 'brand', 'car_series', 'car_model', 'configuration', 'engine_number', 'unnamed:_21', 'color']
expected_columns = list(translated_headers.values())  # Expected columns after translation
chunk_size = 250000  # Adjust chunk_size to 250,000
columns_to_clean = ['email', 'mobile_phone']
pattern = r'[^\w\s@.\-]'

# Function to validate email
def is_valid_email(email):
    """Validates the format of an email address."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

# Function to validate mobile numbers
def is_valid_mobile(mobile):
    """Checks if the mobile number is valid (numeric and reasonable length)."""
    return bool(re.match(r'^\d{7,15}$', str(mobile))) if pd.notna(mobile) else False

# Function to check if VIN and ID are alphanumeric
def is_alphanumeric(value):
    """Checks if a value contains only alphanumeric characters."""
    if pd.isna(value):  # Check for NaN
        return False
    return bool(re.match(r'^[a-zA-Z0-9]+$', str(value)))

# Function to handle errors during data processing
def safe_process(func, *args, **kwargs):
    """Wraps a function in a try-except block for error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return pd.NA

def process_data(file_path, clean_file_prefix, garbage_file_prefix):
    """Processes the dataset in chunks and handles cleaning, validation, and error logging."""
    logging.info(f"Starting data processing for {file_path} with chunk size {chunk_size}")
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Lists to store cleaned and garbage chunks for final merging
    all_cleaned_chunks = []
    all_garbage_chunks = []
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, delimiter=',')):
        logging.info(f"Processing chunk {i+1}")
        
        try:
            # Make a copy to avoid the "Setting on a copy of a slice" warning
            original_chunk = chunk.copy()

            # Translate column names
            chunk.rename(columns=translated_headers, inplace=True)

            # Convert all column names to lowercase and replace spaces with underscores
            chunk.columns = chunk.columns.str.lower().str.replace(" ", "_")

            # Drop the specified columns only from the clean data (keep them in the garbage data)
            clean_chunk = chunk.drop(columns=[col for col in columns_to_remove_in_clean if col in chunk.columns], errors='ignore')

            # Explicitly drop 'unnamed:_21' from clean data if it exists
            if 'unnamed:_21' in clean_chunk.columns:
                clean_chunk.drop('unnamed:_21', axis=1, inplace=True)

            # Convert relevant columns to strings to avoid float issues
            chunk['vin'] = chunk['vin'].astype(str)
            chunk['id_number'] = chunk['id_number'].astype(str)
            chunk['email'] = chunk['email'].astype(str)

            # 1. Check for duplicates based on VIN and ID from the original data
            duplicates = original_chunk.duplicated(subset=['车架号', '身份证'], keep=False)
            chunk['duplicate'] = duplicates

            # 2. Clean the email column
            chunk['email'] = chunk['email'].str.lower().replace('no email', pd.NA)
            chunk['email_valid'] = chunk['email'].apply(is_valid_email)

            # 3. Merge address, city, province, and postal code into a single 'full_address' column
            chunk['full_address'] = chunk[['address', 'city', 'province', 'postal_code']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

            # Keep the original address columns in the garbage files but drop from the clean data
            clean_chunk.drop(['address', 'city', 'province', 'postal_code'], axis=1, inplace=True, errors='ignore')

            # Add 'full_address' column to the clean data
            clean_chunk['full_address'] = chunk['full_address']

            # 4. Check if VIN and ID are alphanumeric
            chunk['vin_valid'] = chunk['vin'].apply(is_alphanumeric)
            chunk['id_number_valid'] = chunk['id_number'].apply(is_alphanumeric)

            # 5. Validate mobile phone and date of birth
            chunk['mobile_valid'] = chunk['mobile_phone'].apply(is_valid_mobile)
            chunk['birthday_valid'] = chunk['date_of_birth'].apply(lambda x: pd.notna(pd.to_datetime(x, errors='coerce')))

            # Drop 'full_address' column from garbage data
            garbage_chunk = chunk.drop(columns=['full_address'], errors='ignore')

            # Combine validation for final clean and garbage datasets
            valid_rows = chunk['vin_valid'] & chunk['id_number_valid'] & chunk['email_valid'] & chunk['mobile_valid'] & chunk['birthday_valid']
            valid_chunk = clean_chunk[valid_rows]
            garbage_chunk = garbage_chunk[~valid_rows | duplicates]  # Include duplicates in garbage
            
            # Log how many rows are garbage
            logging.info(f"Chunk {i+1} has {len(garbage_chunk)} garbage rows (including duplicates).")

            # Save valid chunk
            valid_chunk_file = f"{clean_file_prefix}_chunk_{i+1}.csv"
            valid_chunk.to_csv(valid_chunk_file, index=False)
            logging.info(f"Saved cleaned data chunk {i+1} to {valid_chunk_file}")
            
            # Save garbage chunk (if any)
            if not garbage_chunk.empty:
                garbage_chunk_file = f"{garbage_file_prefix}_chunk_{i+1}.csv"
                garbage_chunk.to_csv(garbage_chunk_file, index=False)
                logging.info(f"Saved garbage data chunk {i+1} to {garbage_chunk_file}")
            else:
                logging.info(f"No garbage data in chunk {i+1}.")
            
            # Collect chunks for final concatenation
            all_cleaned_chunks.append(valid_chunk)
            all_garbage_chunks.append(garbage_chunk)
        
        except Exception as e:
            logging.error(f"Error processing chunk {i+1}: {e}")
    
    logging.info("Data processing complete.")

    # After processing all chunks, concatenate and save final datasets
    if all_cleaned_chunks:
        final_cleaned_data = pd.concat(all_cleaned_chunks, ignore_index=True)
        final_cleaned_file = f"{clean_file_prefix}_final.csv"
        final_cleaned_data.to_csv(final_cleaned_file, index=False)
        logging.info(f"Saved final cleaned dataset to {final_cleaned_file}")

    if all_garbage_chunks:
        final_garbage_data = pd.concat(all_garbage_chunks, ignore_index=True)
        final_garbage_file = f"{garbage_file_prefix}_final.csv"
        final_garbage_data.to_csv(final_garbage_file, index=False)
        logging.info(f"Saved final garbage dataset to {final_garbage_file}")

# Example usage
input_file = r"C:\Users\garne\Documents\DATA CLEANING\760k-Car-Owners-Nationwide-China-csv-2020.csv"
clean_file_prefix = r'C:\Users\garne\Documents\DATA CLEANING\Clean Sets\Clean_China'
garbage_file_prefix = r'C:\Users\garne\Documents\DATA CLEANING\Garbage Sets\Garbage_China'

# Process the data and save chunks
process_data(input_file, clean_file_prefix, garbage_file_prefix)
