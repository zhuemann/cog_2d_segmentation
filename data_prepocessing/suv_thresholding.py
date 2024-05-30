
import pandas as pd

def threshold_suv_max(df):
    # Filter the DataFrame to keep only rows where 'suv' is 2.5 or higher
    df = df[df['SUV'] >= 2.5]
    print(f" After2.5 suv filter {len(df)}")

    return df