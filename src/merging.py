import pandas as pd
import numpy as np

# --- 1. Loading Data ---
# Adjust paths if necessary
files = {
    'aave': "../raw_data/New_source_data/aave_usdc_dados_completos.csv",
    'curve': "../raw_data/New_source_data/curve_3pool_usdc_depeg_risk_completo (1).csv",
    'btc': "../raw_data/features/CBBTCUSD.csv",
    'eth': "../raw_data/features/CBETHUSD.csv",
    'dgs2': "../raw_data/features/DGS2.csv",
    'gas': "../raw_data/features/export-AvgGasPrice.csv",
    'vix': "../raw_data/features/VIXCLS.csv",
    'sofr': "../raw_data/sofr/SOFR.csv"
}

dfs = {}
for name, path in files.items():
    try:
        dfs[name] = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {name}: {e}")

# --- 2. Cleaning Functions ---
def clean_daily(df, date_col, value_col, new_name=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Force numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Rename
    if new_name:
        df = df.rename(columns={value_col: new_name})
        target_col = new_name
    else:
        target_col = value_col
        
    # FIX: Select ONLY the numeric column before resampling
    return df[[target_col]].resample('D').mean()

# --- 3. Processing Specific Files ---

# Aave
df_aave = dfs['aave'].copy()
df_aave['date'] = pd.to_datetime(df_aave['date']).dt.normalize()
df_aave = df_aave.set_index('date').sort_index()
# Grouping explicitly by numeric columns
df_aave = df_aave.groupby('date')[['Y_Aave_APY', 'X_Aave_Utilization', 'totalValueLockedUSD']].mean()

# Curve
df_curve = dfs['curve'].copy()
df_curve['date'] = pd.to_datetime(df_curve['date']).dt.normalize()
df_curve = df_curve.set_index('date').sort_index()
df_curve = df_curve.groupby('date')[['X_USDC_Depeg_Risk']].mean()

# Standard Macro/Crypto
df_btc = clean_daily(dfs['btc'], 'observation_date', 'CBBTCUSD', 'X_BTC_Price')
df_eth = clean_daily(dfs['eth'], 'observation_date', 'CBETHUSD', 'X_ETH_Price')
df_vix = clean_daily(dfs['vix'], 'observation_date', 'VIXCLS', 'X_VIX')
df_sofr = clean_daily(dfs['sofr'], 'observation_date', 'SOFR', 'Y_SOFR')
df_dgs2 = clean_daily(dfs['dgs2'], 'observation_date', 'DGS2', 'X_DGS2')

# Gas (This was the source of the error)
df_gas = dfs['gas'].copy()
df_gas['date'] = pd.to_datetime(df_gas['Date(UTC)'])
df_gas = df_gas.set_index('date').sort_index()
df_gas['X_Gas_Price_Gwei'] = pd.to_numeric(df_gas['Value (Wei)'], errors='coerce') / 1e9
# FIX: Explicit selection [[...]]
df_gas = df_gas[['X_Gas_Price_Gwei']].resample('D').mean()

# --- 4. Merging ---
# Base: Aave + SOFR (Risk Premium components)
df_master = df_aave.join(df_sofr, how='inner')
df_master = df_master.join(df_curve, how='left')
df_master = df_master.join(df_btc, how='left')
df_master = df_master.join(df_eth, how='left')
df_master = df_master.join(df_vix, how='left')
df_master = df_master.join(df_dgs2, how='left')
df_master = df_master.join(df_gas, how='left')

# Forward Fill (Crypto works 24/7, Macro closes on weekends)
df_master = df_master.ffill()

# --- 5. Feature Engineering ---
# Target Spread
df_master['Y_Target_Spread'] = df_master['Y_Aave_APY'] - df_master['Y_SOFR']

# Volatility & Trends
df_master['X_ETH_Returns'] = df_master['X_ETH_Price'].pct_change()
df_master['X_ETH_Vol_7D'] = df_master['X_ETH_Returns'].rolling(window=7).std() * np.sqrt(7)

df_master['X_BTC_Returns'] = df_master['X_BTC_Price'].pct_change()
df_master['X_BTC_Vol_7D'] = df_master['X_BTC_Returns'].rolling(window=7).std() * np.sqrt(7)

df_master['X_Gas_MA_7D'] = df_master['X_Gas_Price_Gwei'].rolling(window=7).mean()
df_master['X_VIX_Pct_Change'] = df_master['X_VIX'].pct_change()

# Final Cleanup
df_final = df_master.dropna()

print("Processing complete.")
print(f"Final Dataset Shape: {df_final.shape}")
df_final.to_csv("../data/dataset_final.csv")
