import pandas as pd
import re
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
expected_columns = ['id', 'login_id', 'mail_address', 'password', 'created_at', 'salt', 'birthday_on', 'gender']
rename_mapping = {
    'ID': 'id',
    'Name': 'login_id',
    'Email': 'mail_address',
    'Date_of_Birth': 'birthday_on',
    'Salary': 'password'
}

# Adjust chunk_size to process four chunks
chunk_size = 900000  
columns_to_clean = ['login_id', 'mail_address']
pattern = r'[^\w\s@.\-]'

# Function to validate email
def is_valid_email(email):
    """Validates the format of a mail address."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

# Function to check for missing critical columns
def is_valid_row(row):
    """Validates a row to ensure no critical columns are missing."""
    if pd.isna(row['login_id']) or pd.isna(row['mail_address']):
        return False  # Consider it garbage if login_id or mail_address is missing
    return True

# Function to check if birthday is a valid date
def is_valid_birthday(birthday):
    """Check if the birthday is a valid date."""
    try:
        pd.to_datetime(birthday, errors='raise')  # Try to convert to datetime
        return True
    except Exception:
        return False

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
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, delimiter=';')):
        logging.info(f"Processing chunk {i+1}")
        
        try:
            # Make a copy to avoid the "Setting on a copy of a slice" warning
            chunk = chunk.copy()

            # Print the column names for debugging
            print(f"Columns in chunk {i+1}: {chunk.columns.tolist()}")

            # Rename columns
            chunk.rename(columns=rename_mapping, inplace=True)

            # Ensure expected columns are present
            for col in expected_columns:
                if col not in chunk.columns:
                    chunk[col] = pd.NA  # or a default value
            
            # Reorder columns
            chunk = chunk[expected_columns]
            
            # Remove empty lines
            chunk.dropna(how='all', inplace=True)
            
            # Remove unauthorized characters from specified columns
            for col in columns_to_clean:
                chunk[col] = safe_process(lambda: chunk[col].astype(str).str.replace(pattern, '', regex=True))
            
            # Standardize data types
            chunk['birthday_on'] = pd.to_datetime(chunk['birthday_on'], errors='coerce')
            
            # Modify 'created_at' to remove the time part
            chunk['created_at'] = pd.to_datetime(chunk['created_at'], errors='coerce').dt.date

            # Initialize garbage validation for multiple criteria
            chunk['Email_Valid'] = chunk['mail_address'].apply(is_valid_email)
            chunk['Row_Valid'] = chunk.apply(is_valid_row, axis=1)  # Ensure no critical columns are missing
            chunk['Birthday_Valid'] = chunk['birthday_on'].apply(is_valid_birthday)  # Check valid birthday
            
            # Identify duplicate rows based on 'login_id' and 'mail_address'
            chunk['Duplicate'] = chunk.duplicated(subset=['login_id', 'mail_address'], keep=False)
            
            # Combine all validation criteria (valid email, valid row, valid birthday, and not a duplicate)
            valid_rows = (chunk['Email_Valid'] & chunk['Row_Valid'] & chunk['Birthday_Valid'] & ~chunk['Duplicate'])
            valid_chunk = chunk[valid_rows].drop(columns=['Email_Valid', 'Row_Valid', 'Birthday_Valid', 'Duplicate'])
            garbage_chunk = chunk[~valid_rows]  # Rows that failed any validation check
            
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

# Main entry point
if __name__ == "__main__":
    # File paths
    input_file = r'C:\Users\garne\Documents\DATA CLEANING\DataSets\3.6M-Japan-lifebear.com-Largest-Notebook-App-UsersDB-csv-2019.csv'  # Update this path to your actual CSV file
    clean_file_prefix = r'C:\Users\garne\Documents\DATA CLEANING\Clean Sets\Clean_Japan'
    garbage_file_prefix = r'C:\Users\garne\Documents\DATA CLEANING\Garbage Sets\Garbage_Japan'
    
    try:
        # Process the data and save chunks
        process_data(input_file, clean_file_prefix, garbage_file_prefix)
        
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        print(f"An error occurred: {e}")
