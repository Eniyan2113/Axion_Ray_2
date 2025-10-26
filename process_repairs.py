import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Step 1: Load Data ---

file_name = 'task2.csv'
print(f"Loading data from '{file_name}'...")

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

print(f"Successfully loaded data. Shape: {df.shape}")
print("-" * 30)

# --- Step 2: Standardize Column Names ---

df_cleaned = df.copy()
df_cleaned.columns = [col.lower().strip().replace(' ', '_') for col in df_cleaned.columns]
print("Column names standardized to snake_case.")

# --- Step 3: Clean NLP and Text Columns ---

print("Cleaning text columns...")

def clean_text(text):
    """
    Cleans text by removing source tags, newlines, and extra whitespace.
    Fixes the '  ' encoding error and converts to lowercase.
    """
    if not isinstance(text, str):
        return text  # Return non-string values (like NaN) as is

    # 1. Remove tags
    text = re.sub(r'\\', '', text)

    # 2. Remove newlines and tabs, replace with a single space
    text = re.sub(r'[\n\t]+', ' ', text)

    # 3. Fix specific known encoding errors
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
    'correction_verbatim', 'customer_verbatim', 'engine_desc',
    'transmission_desc', 'causal_part_nm', 'global_labor_code_description'
]

for col in text_cols_to_clean:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].apply(clean_text)

print("Text cleaning complete.")

# --- Step 4: Correct Data Types ---

print("Correcting data types...")

if 'repair_date' in df_cleaned.columns:
    df_cleaned['repair_date'] = pd.to_datetime(df_cleaned['repair_date'], errors='coerce')

numeric_cols = ['repair_age', 'km', 'reporting_cost', 'totalcost', 'lbrcost']
for col in numeric_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

print("Data types corrected. Conversion errors are now NaN/NaT.")

# --- Step 5: Drop Rows with > 5 Missing Values ---

print("Dropping rows with more than 5 missing values...")
rows_before = df_cleaned.shape[0]
total_cols = df_cleaned.shape[1]
# A row must have at least (total_cols - 5) non-null values to be kept
min_non_null = total_cols - 5

df_cleaned = df_cleaned.dropna(thresh=min_non_null)

rows_after = df_cleaned.shape[0]
print(f"Dropped {rows_before - rows_after} rows for having more than 5 missing values.")

# --- Step 6: Impute Remaining Gaps ("try to fill or leave as is") ---

print("Attempting to fill remaining missing values...")

# 1. Fill numeric columns with their median
for col in numeric_cols:
    if col in df_cleaned.columns:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)

# 2. Fill ALL text/object columns with 'Unknown'
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = df_cleaned[col].fillna('Unknown')

print("Imputation step complete.")

# --- Step 7: Handle Outliers ---

print("Identifying and handling outliers...")
outlier_check_cols = ['km', 'totalcost', 'lbrcost']

for col in outlier_check_cols:
    if col in df_cleaned.columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1 # Corrected IQR calculation

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = df_cleaned[
            (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        ].shape[0]

        print(f"'{col}': Found {outlier_count} potential outliers.")

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
    print("Categorical standardization complete.")
else:
    print("'causal_part_nm' column not found, skipping standardization.")

# --- Step 9: NLP Tag Generation (TF-IDF) ---

print("\n" + "=" * 30)
print("Starting NLP Tag Generation (TF-IDF)...")

# Ensure text columns are strings and fill any potential NaNs with empty string
df_cleaned['customer_verbatim'] = df_cleaned['customer_verbatim'].astype(str).fillna('')
df_cleaned['correction_verbatim'] = df_cleaned['correction_verbatim'].astype(str).fillna('')

# Create a combined "corpus" for the NLP model
df_cleaned['nlp_corpus'] = df_cleaned['customer_verbatim'] + ' ' + df_cleaned['correction_verbatim']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=30, ngram_range=(1, 3))

# Fit the vectorizer to the corpus and transform the text into a matrix
tfidf_matrix = vectorizer.fit_transform(df_cleaned['nlp_corpus'])

# Get the list of feature names (the tags)
nlp_tags_raw = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense DataFrame (needed for consolidation)
# Use raw tags as column names temporarily
df_tags_binary = pd.DataFrame(tfidf_matrix.toarray(), columns=nlp_tags_raw)
df_tags_binary = (df_tags_binary > 0).astype(int)

print(f"Generated {len(nlp_tags_raw)} tags dynamically using NLP.")

# --- Step 10: Create Single, Comma-Separated Tag Column ---

print("Consolidating NLP tags into a single column...")

def join_nlp_tags(row):
    """
    Takes a row of the df_tags_binary, finds all columns
    where the value is 1, and joins the column names
    (the raw tags) into a single string.
    """
    # Find all column names (tags) where the row's value is 1
    active_tags = row[row == 1].index.tolist()

    # Join them with a comma and space
    return ', '.join(active_tags)

# Apply the function across every row (axis=1)
# Use df_tags_binary which has the raw tag names as columns
nlp_tags_consolidated = df_tags_binary.apply(join_nlp_tags, axis=1)

# Add this consolidated Series as a new column to the main cleaned DataFrame
df_cleaned['nlp_tags_consolidated'] = nlp_tags_consolidated

print("Consolidated NLP tag column created.")

# --- Step 11: Create and Save Final Output File (ID + Consolidated NLP Tags Only) ---

print("\n" + "=" * 30)
print("Filtering DataFrame to include only transaction_id and consolidated NLP tags...")

# Define the final list of columns to save
final_cols_to_save = []

# Check if 'transaction_id' exists and add it
if 'transaction_id' in df_cleaned.columns:
    final_cols_to_save.append('transaction_id')
else:
    print("Warning: 'transaction_id' column not found. It will be missing from the output.")

# Add the new consolidated NLP tag column
final_cols_to_save.append('nlp_tags_consolidated')

# Create the final DataFrame subset containing only the desired columns
# Ensure column names exist before subsetting
existing_cols = [col for col in final_cols_to_save if col in df_cleaned.columns]
df_final_output = df_cleaned[existing_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Step 12: Display Sample and Store Final Data ---
print("Sample of Final Output Data:")
print(df_final_output.head(10))

output_file_name = 'transaction_id_with_consolidated_nlp_tags.csv'
# Save the new, smaller DataFrame
df_final_output.to_csv(output_file_name, index=False, encoding='utf-8')

print("\n" + "=" * 30)
print(f"Successfully saved final data (ID and Consolidated NLP Tags only) to: {output_file_name}")