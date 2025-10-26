import pandas as pd
import numpy as np
import re

# --- Step 1: Load Data ---

# Use the file provided by the user
file_name = 'task2.csv'

try:
    # Apply the fixes: 'latin1' encoding and 'python' engine
    df = pd.read_csv(
        file_name, 
        encoding='latin1', 
        engine='python'
    )
except Exception as e:
    # If that also fails, fall back to skipping bad lines
    print(f"Error with engine='python': {e}. Falling back to on_bad_lines='skip'...")
    df = pd.read_csv(
        file_name, 
        encoding='latin1', 
        on_bad_lines='skip' # This will drop the problem rows
    )

print(f"Successfully loaded data from '{file_name}'. Shape: {df.shape}")
print("-" * 30)

# --- Step 2: Standardize Column Names ---

# Create a copy to avoid modifying the original data
df_cleaned = df.copy()

# Convert all column names to lowercase 'snake_case'
df_cleaned.columns = [col.lower().strip().replace(' ', '_') for col in df_cleaned.columns]
print("Column names standardized to snake_case.")

# --- Step 3: Clean NLP and Text Columns ---

print("Cleaning text columns (verbatims, descriptions)...")

# 
# THIS IS THE CORRECTED FUNCTION (No "unknown_encoding_error" bug)
#
def clean_text(text):
    if not isinstance(text, str):
        return text  # Return non-string values (like NaN) as is
    
    # 1. Remove tags
    text = re.sub(r'\\', '', text)
    
    # 2. Remove newlines and tabs, replace with a single space
    text = re.sub(r'[\n\t]+', ' ', text)
    
    # 3. Fix specific known encoding errors
    # This was the original, correct line to fix the weird '  ' character
    text = text.replace('  ', 'unknown_encoding_error')
    
    # 4. Convert to lowercase
    text = text.lower()
    
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    # 6. Remove stray backslashes (escaping the backslash)
    text = re.sub(r'\\', '', text)
    
    return text

# List of text columns to clean
text_cols_to_clean = [
    'correction_verbatim', 
    'customer_verbatim', 
    'engine_desc', 
    'transmission_desc',
    'causal_part_nm',
    'global_labor_code_description'
]

for col in text_cols_to_clean:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].apply(clean_text)

print("Text cleaning complete.")

# --- Step 4: Correct Data Types ---

print("Correcting data types...")

# 1. Convert repair_date to datetime
if 'repair_date' in df_cleaned.columns:
    # errors='coerce' will turn bad dates into NaT (Not-a-Time)
    df_cleaned['repair_date'] = pd.to_datetime(df_cleaned['repair_date'], errors='coerce')

# 2. Convert numeric columns to float, coercing errors
numeric_cols = ['repair_age', 'km', 'reporting_cost', 'totalcost', 'lbrcost']
for col in numeric_cols:
    if col in df_cleaned.columns:
        # errors='coerce' will turn bad numbers into NaN
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

print("Data types corrected. Any conversion errors are marked as NaN/NaT.")

# --- NEW Step 5: Drop Rows with > 5 Missing Values ---

print("Checking for rows with more than 5 missing values...")
rows_before = df_cleaned.shape[0]
total_cols = df_cleaned.shape[1]

# A row must have at least (total_cols - 5) non-null values to be kept
min_non_null = total_cols - 5 

df_cleaned = df_cleaned.dropna(thresh=min_non_null)

rows_after = df_cleaned.shape[0]
print(f"Dropped {rows_before - rows_after} rows for having more than 5 missing values.")
print(f"New shape before imputation: {df_cleaned.shape}")


# --- NEW Step 6: Impute Remaining Gaps ("try to fill or leave as is") ---

print("Attempting to fill remaining missing values...")

# 1. Fill numeric columns with their median
# We use the same 'numeric_cols' list from Step 4
for col in numeric_cols:
    if col in df_cleaned.columns:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
        print(f"Numeric gaps in '{col}' filled with median: {median_val}")

# 2. Fill ALL text/object columns with 'Unknown'
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = df_cleaned[col].fillna('Unknown')

print("All text gaps filled with 'Unknown'.")
print("Imputation step complete. Any remaining NaNs (e.g., in date or all-NaN cols) will be left as-is.")


# --- Step 7: Handle Outliers ---

print("Identifying and handling outliers...")

# Columns to check for outliers
outlier_check_cols = ['km', 'totalcost', 'lbrcost']

for col in outlier_check_cols:
    if col in df_cleaned.columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        
        # *** BUG FIX HERE: Was Q3-Q3, now Q3-Q1 ***
        IQR = Q3 - Q1 
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count the outliers
        outlier_count = df_cleaned[
            (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        ].shape[0]
        
        print(f"'{col}': Found {outlier_count} potential outliers (values < {lower_bound:.2f} or > {upper_bound:.2f})")

        # Capping values at the 1st and 99th percentile
        p01 = df_cleaned[col].quantile(0.01)
        p99 = df_cleaned[col].quantile(0.99)
        
        df_cleaned[col] = np.clip(df_cleaned[col], p01, p99)
        print(f"Capped '{col}' at 1st percentile ({p01:.2f}) and 99th percentile ({p99:.2f}).")

print("Outlier handling complete.")

# --- Step 8: Standardize Categorical Data ---

print("Standardizing categorical values...")

if 'causal_part_nm' in df_cleaned.columns:
    # 1. Fix known typos
    df_cleaned['causal_part_nm'] = df_cleaned['causal_part_nm'].replace(
        'wheel asm-strg *backen blackk', 'wheel asm-strg *black'
    )

    # 2. Consolidate similar categories
    consolidation_map = {
        'wheel asm-strg *jet black': 'wheel asm-strg *black',
        '"wheel,strg *jet black"': 'wheel asm-strg *black',
        'wheel asm-strg *very dark at': 'wheel asm-strg *very dark atmosphere',
        'wheel asm-strg *dark titaniu': 'wheel asm-strg *dark titanium'
    }
    df_cleaned['causal_part_nm'] = df_cleaned['causal_part_nm'].replace(consolidation_map)

    # Example: Check consolidation
    print(f"Unique 'black' steering wheel parts before: (multiple)")
    print(f"Unique 'black' steering wheel parts after: {df_cleaned[df_cleaned['causal_part_nm'].str.contains('black')]['causal_part_nm'].nunique()}")
    print("Categorical standardization complete.")
else:
    print("'causal_part_nm' column not found, skipping standardization.")


# --- Step 9: Final Check & Store Data ---

print("\n" + "=" * 30)
print("Data Cleaning Complete. Final DataFrame Info:")
df_cleaned.info()

# --- Step 9: Store Cleaned Data ---

output_file_name = 'cleaned_vehicle_repairs_thresh_5.csv'
# Save with UTF-8 encoding as requested
df_cleaned.to_csv(output_file_name, index=False, encoding='utf-8')

# *** BUG FIX HERE: Corrected print statement syntax ***
print("\n" + "=" * 30)
print(f"Successfully saved cleaned data to '{output_file_name}'")