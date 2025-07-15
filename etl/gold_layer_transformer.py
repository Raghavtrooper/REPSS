import pandas as pd
import numpy as np
import re # For phone number cleaning
import uuid # For generating unique group IDs

def transform_to_gold_layer(structured_profiles: list) -> pd.DataFrame:
    """
    Transforms a list of silver-layer structured employee profiles into a gold-layer
    Pandas DataFrame, performing cleaning and normalization, and ANNOTATING duplicates
    rather than dropping or merging them.
    Updated to handle new 'location', 'objective', 'qualifications_summary',
    'experience_summary', 'has_photo' fields, and the new 'companies_worked_with_duration' field.

    Note: As per the updated ETL pipeline, the output of this function is now primarily
    for separate storage (e.g., Parquet in MinIO) and analysis, and its output is used
    as the payload for Qdrant embeddings.
    """
    print("\nStarting gold layer transformation (annotating duplicates)..")
    if not structured_profiles:
        print("No structured profiles to transform. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(structured_profiles)

    # Ensure all expected top-level columns are present, adding them with a default if missing
    expected_string_cols = ['name', 'location', 'objective', 'email_id', 'phone_number',
                            'experience_summary', 'qualifications_summary',
                            'current_job_title', 'linkedin_url', 'github_url'
                           ]
    for col in expected_string_cols:
        if col not in df.columns:
            df[col] = "Unknown" # Default to "Unknown" if the column is entirely missing from input

    # 1. Initial Cleaning & Normalization (applies to individual records)
    # No 'title' or 'department' cleaning needed as they are no longer extracted

    print("  Normalizing skill names...")
    if 'skills' in df.columns:
        df['skills'] = df['skills'].apply(lambda x: [s.lower().strip() for s in x] if isinstance(x, list) else ([str(x).lower().strip()] if isinstance(x, str) and str(x).strip() and str(x).lower() != "not found" else []))
        
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


    # Fill missing fields to ensure consistent grouping keys and output
    print("  Filling missing fields for consistent grouping...")
    
    # Ensure 'id' column is present and cleaned
    if 'id' in df.columns:
        missing_id_mask = df['id'].isna() | \
                          (df['id'].astype(str).str.strip().str.lower() == 'not found') | \
                          (df['id'].astype(str).str.strip() == '')
        if missing_id_mask.any():
            df.loc[missing_id_mask, 'id'] = df.loc[missing_id_mask].index.astype(str) + '_generated_id'
        df['id'] = df['id'].astype(str).str.strip() # Ensure all IDs are string and stripped
    else:
        df['id'] = df.index.astype(str) + '_generated_id'
        print("    'id' column not found, generated new IDs based on index.")

    # Fill remaining string columns with "Unknown" for consistency
    string_cols_to_fill = ['name', 'location', 'objective', 'email_id', 'phone_number',
                           'experience_summary', 'qualifications_summary', 'current_job_title',
                           'linkedin_url', 'github_url'] # Updated list of string columns
    for col in string_cols_to_fill:
        if col in df.columns:
            if df[col].isnull().any() or (df[col].astype(str).str.lower() == 'not found').any() or (df[col].astype(str).str.strip() == '').any():
                df[col] = df[col].replace('Not Found', np.nan).fillna("Unknown").astype(str).str.strip()
            else: # Ensure type is string even if no NaNs/unknowns
                df[col] = df[col].astype(str).str.strip()

    # Ensure 'has_photo' is a boolean
    if 'has_photo' in df.columns:
        df['has_photo'] = df['has_photo'].fillna(False).astype(bool)
    else:
        df['has_photo'] = False # Default to False if column is missing

    # Ensure 'companies_worked_with_duration' is a list of strings, default to empty list
    if 'companies_worked_with_duration' in df.columns:
        # Convert any non-list values to empty list or ensure it's a list of strings
        df['companies_worked_with_duration'] = df['companies_worked_with_duration'].apply(
            lambda x: [str(s).strip() for s in x if str(s).strip()] if isinstance(x, list) else []
        )
    else:
        df['companies_worked_with_duration'] = [[] for _ in range(len(df))] # Add as empty lists if missing

    # Handle 'certifications' column
    print("  Ensuring 'certifications' column is a list of strings...")
    if 'certifications' in df.columns:
        df['certifications'] = df['certifications'].apply(
            lambda x: [str(s).strip() for s in x if str(s).strip()] if isinstance(x, list) else []
        )
    else:
        df['certifications'] = [[] for _ in range(len(df))] # Add as empty lists if missing

    # Handle 'projects' column
    print("  Ensuring 'projects' column is a list of strings...")
    if 'projects' in df.columns:
        df['projects'] = df['projects'].apply(
            lambda x: [str(s).strip() for s in x if str(s).strip()] if isinstance(x, list) else []
        )
    else:
        df['projects'] = [[] for _ in range(len(df))] # Add as empty lists if missing


    # 2. Identify and Annotate Duplicate Profiles
    print("  Identifying and annotating duplicate profiles...")
    
    # Create primary and secondary grouping keys
    # Use '_raw_email_key' to ensure 'Unknown' is treated as a distinct group for grouping
    df['_primary_dedupe_key'] = df['email_id'].apply(lambda x: x if x and x != 'Unknown' else np.nan) # Use np.nan for unknown emails
    df['_secondary_dedupe_key'] = df['name'] + '__' + df['phone_number'] # Combine name and phone for secondary


    # Sort to ensure consistent "master" record selection for each group
    # Prioritize valid email_id, then name, then original_filename
    df.sort_values(by=['_primary_dedupe_key', '_secondary_dedupe_key', '_original_filename'], na_position='last', inplace=True)
    df = df.reset_index(drop=True) # Reset index after sorting to avoid issues with loc/iloc

    # Initialize new annotation columns
    df['_is_master_record'] = False
    df['_duplicate_group_id'] = None # Will store UUID for the group
    df['_duplicate_count'] = 1 # Default to 1 (itself) for unique records
    df['_associated_original_filenames'] = None # Will store list of all filenames in a group
    df['_associated_ids'] = None # Will store list of all original 'id's in a group
    
    processed_indices = set() # To keep track of rows already assigned to a group ID

    # Group by the most reliable identifier first: valid email_id
    email_groups = df.groupby('_primary_dedupe_key', dropna=False) # Keep NaN groups
    
    for email_key, group_by_email in email_groups:
        if pd.isna(email_key): # This group contains profiles with unknown/missing email_id
            # For records with unknown/missing email, group further by secondary key (name+phone)
            secondary_groups = group_by_email.groupby('_secondary_dedupe_key', dropna=False) # Keep NaN groups
            for sec_key, group_by_sec_key in secondary_groups:
                # Only process if this group hasn't been processed via a different path already
                if not group_by_sec_key.index.difference(list(processed_indices)).empty:
                    master_idx = group_by_sec_key.index[0] # The first record in this sub-group will be the master
                    group_id = str(uuid.uuid4())
                    
                    df.loc[master_idx, '_is_master_record'] = True
                    df.loc[group_by_sec_key.index, '_duplicate_group_id'] = group_id
                    df.loc[master_idx, '_duplicate_count'] = len(group_by_sec_key)
                    df.at[master_idx, '_associated_original_filenames'] = group_by_sec_key['_original_filename'].tolist() # Fix: Use .at for single cell assignment
                    df.at[master_idx, '_associated_ids'] = group_by_sec_key['id'].tolist() # Fix: Use .at for single cell assignment
                    processed_indices.update(group_by_sec_key.index)
        else: # This group contains profiles with a known email_id
            # Only process if this group hasn't been processed via a different path already
            if not group_by_email.index.difference(list(processed_indices)).empty:
                master_idx = group_by_email.index[0] # The first record in this group will be the master
                group_id = str(uuid.uuid4())
                
                df.loc[master_idx, '_is_master_record'] = True
                df.loc[group_by_email.index, '_duplicate_group_id'] = group_id
                df.loc[master_idx, '_duplicate_count'] = len(group_by_email)
                df.at[master_idx, '_associated_original_filenames'] = group_by_email['_original_filename'].tolist() # Fix: Use .at for single cell assignment
                df.at[master_idx, '_associated_ids'] = group_by_email['id'].tolist() # Fix: Use .at for single cell assignment
                processed_indices.update(group_by_email.index)
    
    # Final pass to ensure any remaining un-grouped records get a unique group ID and are marked as master
    for idx in df.index:
        if idx not in processed_indices:
            df.loc[idx, '_duplicate_group_id'] = str(uuid.uuid4())
            df.loc[idx, '_is_master_record'] = True
            df.loc[idx, '_duplicate_count'] = 1 # Explicitly set for single records
            df.loc[idx, '_associated_original_filenames'] = [df.loc[idx, '_original_filename']]
            df.loc[idx, '_associated_ids'] = [df.loc[idx, 'id']]


    # Drop temporary grouping keys
    df.drop(columns=['_primary_dedupe_key', '_secondary_dedupe_key'], errors='ignore', inplace=True)

    print(f"  Annotation complete. Total records: {len(df)}.")

    # 3. Enrich (Optional - Examples)
    # Removed 'job_level' assignment as 'title' is no longer extracted.
    print("  No optional enrichment (e.g., job level tags) performed as 'title' is not extracted.")

    # Remove old columns if they exist from previous ETL runs
    df.drop(columns=['contact_info', 'title', 'department', 'experience', 'education'], errors='ignore', inplace=True)

    print("Gold layer transformation complete.")
    return df