import pandas as pd
import numpy as np
import re # For phone number cleaning

def transform_to_gold_layer(structured_profiles: list) -> pd.DataFrame:
    """
    Transforms a list of silver-layer structured employee profiles into a gold-layer
    Pandas DataFrame, performing cleaning and normalization, and updated for
    'email_id' and 'phone_number' fields.
    """
    print("\nStarting gold layer transformation...")
    if not structured_profiles:
        print("No structured profiles to transform. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(structured_profiles)

    # 1. Clean & Normalize
    print("  Standardizing job titles...")
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).str.lower().str.strip().replace('not found', np.nan)
    
    print("  Normalizing skill names...")
    if 'skills' in df.columns:
        df['skills'] = df['skills'].apply(lambda x: [s.lower().strip() for s in x] if isinstance(x, list) else ([str(x).lower().strip()] if isinstance(x, str) and x.strip() and x != "Not Found" else []))
        
        skill_normalization_map = {
            'python3': 'python', 'aws cloud': 'aws', 'ml': 'machine learning',
            'ai': 'artificial intelligence', 'js': 'javascript', 'c++': 'cpp', 'c#': 'csharp',
        }
        
        def normalize_skill_list(skills):
            if not skills: return []
            normalized_skills = [skill_normalization_map.get(skill, skill) for skill in skills]
            return sorted(list(set(normalized_skills)))

        df['skills'] = df['skills'].apply(normalize_skill_list)

    print("  Cleaning email IDs and phone numbers...")
    if 'email_id' in df.columns:
        df['email_id'] = df['email_id'].astype(str).str.lower().str.strip().replace('not found', np.nan)
        df['email_id'] = df['email_id'].apply(lambda x: x if pd.notna(x) and '@' in str(x) and '.' in str(x) else np.nan)

    if 'phone_number' in df.columns:
        df['phone_number'] = df['phone_number'].astype(str).str.strip().replace('not found', np.nan)
        df['phone_number'] = df['phone_number'].apply(lambda x: re.sub(r'[^\d+]', '', str(x)) if pd.notna(x) else np.nan)
        df['phone_number'] = df['phone_number'].replace('', np.nan) # Replace empty strings with NaN


    print("  Filling missing fields...")
    # Fill None/NaN values with "Unknown" for string fields or appropriate defaults
    string_cols = ['name', 'department', 'experience', 'education', 'email_id', 'phone_number'] # Include new contact fields
    for col in string_cols:
        if col in df.columns:
            # Only fill if the column has NaN/None values, otherwise it might convert valid data
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            else: # Ensure type is string even if no NaNs
                df[col] = df[col].astype(str).str.strip()


    # Fix: Correctly assign fallback ID for missing 'id' values
    if 'id' in df.columns:
        # Create a Series of fallback IDs only for the rows where 'id' is currently missing
        missing_id_mask = df['id'].isna()
        if missing_id_mask.any():
            df.loc[missing_id_mask, 'id'] = df.loc[missing_id_mask].index.astype(str) + '_generated_id'
        df['id'] = df['id'].astype(str).str.strip() # Ensure all IDs are string and stripped
    
    # NEW: Deduplicate based on email_id primarily
    print("  Deduplicating profiles (prioritizing email_id for uniqueness)...")
    initial_rows = len(df)
    
    # First, handle duplicates where email_id is present and valid
    # Ensure 'email_id' column exists and has non-NA values before attempting
    if 'email_id' in df.columns and df['email_id'].notna().any():
        # Temporarily filter out 'Unknown' or NaN emails for this primary deduplication pass
        df_with_email = df[df['email_id'].notna() & (df['email_id'] != 'unknown')].copy()
        df_no_email = df[df['email_id'].isna() | (df['email_id'] == 'unknown')].copy()

        if not df_with_email.empty:
            df_with_email.sort_values(by='email_id', inplace=True) # Optional: sort for consistent 'first' pick
            df_with_email.drop_duplicates(subset=['email_id'], keep='first', inplace=True)
            print(f"  Deduplicated by valid email_id: {len(df_with_email)} records remaining.")
        else:
            print("  No valid email IDs found for primary deduplication.")
        
        # Then, for remaining records without a unique email_id, use a combination of name and original filename
        # This catches cases where email might be missing or generic
        if not df_no_email.empty:
            df_no_email.drop_duplicates(subset=['name', '_original_filename'], keep='first', inplace=True)
            print(f"  Deduplicated remaining records by name/filename: {len(df_no_email)} records.")
        
        # Concatenate back the two dataframes
        df = pd.concat([df_with_email, df_no_email]).reset_index(drop=True)

    else: # Fallback if no email_id column or no non-NA emails at all
        print("  No email_id column or no valid email IDs present. Deduplicating by name and original filename.")
        df.drop_duplicates(subset=['name', '_original_filename'], keep='first', inplace=True)
    
    print(f"  Final deduplication complete. {len(df)} unique profiles remain (from {initial_rows} initial).")


    # 2. Enrich (Optional - Examples)
    print("  Adding optional enrichment (e.g., job level tags)...")
    if 'title' in df.columns:
        def assign_job_level(title):
            if pd.isna(title) or title == "Unknown": return 'Unknown' # Handle explicit "Unknown" from fillna
            title = str(title).lower()
            if 'senior' in title or 'lead' in title or 'principal' in title: return 'Senior'
            elif 'junior' in title or 'associate' in title or 'entry' in title: return 'Junior'
            else: return 'Mid-level'
        df['job_level'] = df['title'].apply(assign_job_level)

    # Remove the old 'contact_info' column if it exists from previous ETL runs
    df.drop(columns=['contact_info'], errors='ignore', inplace=True)

    print("Gold layer transformation complete.")
    return df
