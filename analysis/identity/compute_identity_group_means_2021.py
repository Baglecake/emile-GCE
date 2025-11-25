"""
Compute CES 2021 identity vectors and group-level means.

Input:
  - data/CES_2021.dta (using .dta as parquet was incomplete)
  - data/identity/identity_weights_2021.v0.json

Output:
  - data/identity/identity_group_means_2021.csv
"""
import pandas as pd
import json
import numpy as np
import os

# Paths
# Assuming script is run from project root or paths are absolute
# Using absolute paths for safety as per previous context, but relative to project root is better if possible.
# Sticking to absolute for now to match previous pattern, but updating to new locations.
PROJECT_ROOT = '/Users/delcoburn/Desktop/emile-GCE'
dta_path = os.path.join(PROJECT_ROOT, 'CES_2021.dta')
mapping_path = os.path.join(PROJECT_ROOT, 'data/identity/identity_weights_2021.v0.json')
output_csv_path = os.path.join(PROJECT_ROOT, 'data/identity/identity_group_means_2021.csv')

print("--- 1. Loading Data and Mapping ---")
# Load mapping
with open(mapping_path, 'r') as f:
    mapping = json.load(f)

# Load data
df = pd.read_stata(dta_path, convert_categoricals=False)
print(f"Loaded CES 2021 data: {df.shape}")

# Pre-calculate min/max for normalization
# We need to know the scale of each variable. 
# Since we don't have the codebook metadata easily accessible for range (0-10, 0-100, etc.),
# we will infer it from the data itself (min/max).
# WARNING: This assumes the data contains the full range. 
# For survey data like 0-10, this is usually safe if N is large.

var_stats = {}
all_vars = []
for dim, vars_list in mapping.items():
    for v_obj in vars_list:
        v = v_obj['variable']
        all_vars.append(v)
        if v in df.columns:
            # Convert to numeric, coercing errors
            df[v] = pd.to_numeric(df[v], errors='coerce')
            var_stats[v] = {
                'min': df[v].min(),
                'max': df[v].max()
            }
        else:
            print(f"Warning: Variable {v} not found in dataframe.")

print("Variable stats calculated for normalization.")

def compute_identity_vector(row, mapping, stats):
    """
    Computes identity vector for a single row.
    """
    result = {}
    
    for dim, vars_list in mapping.items():
        dim_val = 0.0
        total_weight = 0.0
        
        for v_obj in vars_list:
            v = v_obj['variable']
            weight = v_obj['weight']
            
            if v not in row or pd.isna(row[v]):
                continue
                
            # Normalize to [0, 1]
            val = row[v]
            v_min = stats[v]['min']
            v_max = stats[v]['max']
            
            if v_max > v_min:
                norm_val = (val - v_min) / (v_max - v_min)
            else:
                norm_val = 0.5 # Fallback if no variance
                
            dim_val += norm_val * weight
            total_weight += weight
            
        if total_weight > 0:
            result[dim] = dim_val / total_weight # Weighted average
        else:
            result[dim] = 0.5 # Default neutral if missing all data
            
    return result

print("\n--- 2. Computing Identity Vectors (Sample) ---")
sample_indices = df.head(10).index
sample_vectors = []

for idx in sample_indices:
    row = df.loc[idx]
    vec = compute_identity_vector(row, mapping, var_stats)
    sample_vectors.append(vec)
    print(f"Row {idx}: {vec}")

print("\n--- 3. Computing for All Respondents ---")
# Vectorized approach for speed
# We'll create new columns for each dimension
for dim, vars_list in mapping.items():
    df[dim] = 0.0
    weight_sum = pd.Series(0.0, index=df.index)
    
    for v_obj in vars_list:
        v = v_obj['variable']
        weight = v_obj['weight']
        
        if v not in df.columns:
            continue
            
        # Normalize
        v_min = var_stats[v]['min']
        v_max = var_stats[v]['max']
        
        if v_max > v_min:
            norm_col = (df[v] - v_min) / (v_max - v_min)
        else:
            norm_col = pd.Series(0.5, index=df.index)
            
        # Add to weighted sum, handling NaNs (fillna(0) for addition, but track weights)
        # Actually, if NaN, we shouldn't add weight.
        not_na = df[v].notna()
        
        df.loc[not_na, dim] += norm_col[not_na] * weight
        weight_sum.loc[not_na] += weight
        
    # Final division
    # Avoid division by zero
    mask = weight_sum > 0
    df.loc[mask, dim] = df.loc[mask, dim] / weight_sum.loc[mask]
    df.loc[~mask, dim] = 0.5 # Default

print("Identity vectors computed.")

print("\n--- 4. Group Summaries ---")
# Grouping variables: Region, pes21_rural_urban, cps21_household
# We need to clean these up first as they might be codes or strings.
# Since we loaded with convert_categoricals=False, they are likely codes or strings.
# Let's check unique values for a few to see if we need mapping (we can just group by whatever is there for now).

group_cols = ['Region', 'pes21_rural_urban', 'cps21_household']
existing_group_cols = [c for c in group_cols if c in df.columns]

print(f"Grouping by: {existing_group_cols}")

if existing_group_cols:
    grouped = df.groupby(existing_group_cols)[['engagement', 'institutional_faith', 'social_friction']].agg(['mean', 'count'])
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        'engagement_mean': 'mean_engagement',
        'engagement_count': 'N',
        'institutional_faith_mean': 'mean_institutional_faith',
        'social_friction_mean': 'mean_social_friction'
    })
    
    # Keep only relevant columns (N is repeated)
    grouped = grouped[['mean_engagement', 'mean_institutional_faith', 'mean_social_friction', 'N']]
    # N is actually engagement_count, which is fine as valid N for that dim. 
    # But let's just take one N column.
    grouped = grouped.loc[:, ~grouped.columns.duplicated()]
    
    print("\nGroup Summary (First 10):")
    print(grouped.head(10))
    
    grouped.to_csv(output_csv_path)
    print(f"\nSaved group summaries to {output_csv_path}")
else:
    print("No grouping variables found.")
