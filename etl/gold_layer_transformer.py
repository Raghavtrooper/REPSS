import pandas as pd
import numpy as np # For potential NaN handling if needed

def transform_to_gold_layer(structured_profiles: list) -> pd.DataFrame:
    """
    Transforms a list of silver-layer structured employee profiles into a gold-layer
    Pandas DataFrame, performing cleaning and normalization.
    
    Args:
        structured_profiles: A list of dictionaries, where each dictionary
                             represents a silver-layer employee profile.
                             
    Returns:
        A Pandas DataFrame representing the cleaned and normalized gold-layer data.
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
        # Optional: More advanced title normalization (e.g., mapping variations to a canonical form)
        # For example:
        # title_mapping = {'software engineer': 'Software Engineer', 'sw engineer': 'Software Engineer'}
        # df['title'] = df['title'].replace(title_mapping)

    print("  Normalizing skill names...")
    if 'skills' in df.columns:
        # Ensure 'skills' is a list, even if it's currently a string or None
        df['skills'] = df['skills'].apply(lambda x: [s.lower().strip() for s in x] if isinstance(x, list) else ([str(x).lower().strip()] if isinstance(x, str) and x.strip() and x != "Not Found" else []))
        
        # Example skill normalization mapping
        skill_normalization_map = {
            'python3': 'python',
            'aws cloud': 'aws',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'js': 'javascript',
            'c++': 'cpp', # Standardizing common abbreviations
            'c#': 'csharp',
        }
        
        def normalize_skill_list(skills):
            if not skills:
                return []
            normalized_skills = []
            for skill in skills:
                # Apply direct mapping, otherwise keep original
                normalized_skills.append(skill_normalization_map.get(skill, skill))
            return sorted(list(set(normalized_skills))) # Deduplicate and sort

        df['skills'] = df['skills'].apply(normalize_skill_list)

    print("  Filling missing fields...")
    # Fill None/NaN values with "Unknown" for string fields or appropriate defaults
    string_cols = ['name', 'department', 'experience', 'education', 'contact_info']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].replace('Not Found', np.nan).fillna("Unknown").astype(str).str.strip()

    # Special handling for numerical or list fields if they could be missing
    if 'id' in df.columns:
        df['id'] = df['id'].fillna("Unknown_ID").astype(str)

    print("  Deduplicating profiles...")
    # Deduplicate based on a combination of fields that uniquely identify a person
    # Using 'id' if it's reliable, otherwise a combination of 'name' and 'contact_info'
    if 'id' in df.columns and not df['id'].astype(str).str.contains('Unknown_ID').any():
        df.drop_duplicates(subset=['id'], inplace=True)
    else:
        # Fallback for deduplication if 'id' is not unique or present
        # This assumes a person can be uniquely identified by name and contact info
        df.drop_duplicates(subset=['name', 'contact_info'], inplace=True, keep='first')
    
    print(f"  Deduplication complete. {len(df)} unique profiles remain.")

    # 2. Enrich (Optional - Examples)
    print("  Adding optional enrichment (e.g., job level tags)...")
    if 'title' in df.columns:
        def assign_job_level(title):
            if pd.isna(title):
                return 'Unknown'
            title = str(title).lower()
            if 'senior' in title or 'lead' in title or 'principal' in title:
                return 'Senior'
            elif 'junior' in title or 'associate' in title or 'entry' in title:
                return 'Junior'
            else:
                return 'Mid-level'
        df['job_level'] = df['title'].apply(assign_job_level)

    print("Gold layer transformation complete.")
    return df
