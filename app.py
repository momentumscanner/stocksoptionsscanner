import streamlit as st
import pandas as pd
import requests
import datetime
import time
import os
import gzip
import json

import concurrent.futures

# --- Helpers ---
def get_ist_now():
    """Get current time in IST (UTC+5:30)"""
    return datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)

# --- Configuration ---
st.set_page_config(page_title="Live Option Scanner", layout="wide")

# Update time for header
update_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-top: -20px; margin-bottom: 10px;'>
        <h3 style='margin: 0;'>Live Option Scanner</h3>
        <span style='font-size: 1rem; color: #555;'>Last Updated: {update_time} (IST)</span>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Try to get token from Secrets (Shared Mode) or Sidebar (Personal Mode)
# Secrets are set in Streamlit Cloud Dashboard
shared_token = None
try:
    if "UPSTOX_TOKEN" in st.secrets:
        shared_token = st.secrets["UPSTOX_TOKEN"]
except FileNotFoundError:
    pass

if shared_token:
    access_token = shared_token
    st.sidebar.success("✅ Shared Access Token Loaded")
    # Optional: Allow override if needed, or just hide input
    # override_token = st.sidebar.text_input("Override Token (Optional)", type="password")
    # if override_token:
    #     access_token = override_token
else:
    access_token = st.sidebar.text_input("Access Token", type="password")

if not access_token:
    st.warning("Please enter your Access Token in the sidebar to proceed.")
    st.stop()

# --- Instruments Data Synchronization ---
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
INSTRUMENTS_FILE = 'NSE.json'

def is_file_fresh(filepath):
    """Check if file exists and is from today"""
    if not os.path.exists(filepath):
        return False
    file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_time.date() == datetime.date.today()

def download_and_extract_instruments():
    """Download and unzip instruments file to NSE.json"""
    status_placeholder = st.empty()
    status_placeholder.info("Downloading latest NSE.json from Upstox...")
    
    try:
        # Download
        response = requests.get(INSTRUMENTS_URL, stream=True)
        response.raise_for_status()
        
        status_placeholder.info("Extracting NSE.json...")
        
        # Decompress and Save to File
        with gzip.GzipFile(fileobj=response.raw) as gz:
            with open(INSTRUMENTS_FILE, 'wb') as f_out:
                while True:
                    chunk = gz.read(1024*1024) # Read in chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    
        status_placeholder.success("NSE.json updated successfully!")
        time.sleep(1)
        status_placeholder.empty()
        return True
        
    except Exception as e:
        status_placeholder.error(f"Failed to update instruments: {e}")
        return False

# --- Data Loading ---
@st.cache_data(ttl=3600*4)  # Cache for 4 hours
def load_data():
    # Check freshness and download if needed
    if not is_file_fresh(INSTRUMENTS_FILE):
        if not download_and_extract_instruments():
             # If download failed, try to use existing file
             if not os.path.exists(INSTRUMENTS_FILE):
                 return pd.DataFrame(), pd.DataFrame()

    # Load and Filter NSE.json directly
    try:
        # We need to filter WHILE loading to save memory if possible, 
        # but standard json.load reads all. 
        # So we load, filter immediately, then delete raw.
        with open(INSTRUMENTS_FILE, 'r') as f:
            data = json.load(f)
            
        # DEBUG: Print data stats
        print(f"DEBUG: Loaded {len(data)} records from NSE.json")
        unique_segments = set(row.get('segment') for row in data[:1000]) # Check first 1000
        print(f"DEBUG: Sample segments: {unique_segments}")

        # Optimize: Filter list before creating DataFrame
        filtered_data = [
            row for row in data 
            if row.get('segment') == 'NSE_FO' and row.get('asset_type') in ['EQUITY', 'INDEX']
        ]
        
        print(f"DEBUG: Filtered down to {len(filtered_data)} records")

        if not filtered_data:
             st.error(f"No NSE_FO data found in NSE.json! Total records: {len(data)}. Segments found: {list(set(r.get('segment') for r in data[:5000]))}")
             return pd.DataFrame(), pd.DataFrame()

        del data # Free huge memory immediately

        # Convert to DataFrame
        df = pd.DataFrame(filtered_data)
        del filtered_data # Free list memory
        
        # 1. Options DF (All NSE_FO EQUITY/INDEX)
        # This is used for looking up CE/PE
        options_df = df.copy()
        
        # 2. Futures DF (Current Month FUT)
        # Filter for FUT
        df_fut = df[df['instrument_type'].str.contains('FUT', na=False)].copy()
        
        # Filter for Near Month Expiry (Nearest valid expiry >= Today)
        if 'expiry' in df_fut.columns:
            # Convert expiry from milliseconds to datetime for filtering
            df_fut['expiry_dt'] = pd.to_datetime(df_fut['expiry'], unit='ms')
            
            # Use IST date for comparison
            current_date = get_ist_now().date()
            
            # Filter futures that haven't expired yet
            # We want expiry >= today (or strictly > today if expired yesterday)
            # Upstox removes expired contracts from NSE.json usually, but let's be safe.
            active_futures = df_fut[df_fut['expiry_dt'].dt.date >= current_date]
            
            if not active_futures.empty:
                # Find the nearest expiry date across ALL active futures
                nearest_expiry = active_futures['expiry_dt'].min()
                print(f"DEBUG: Found nearest expiry: {nearest_expiry}")
                
                # Filter only futures matching this nearest expiry (e.g., Feb if Jan is gone)
                df_fut = active_futures[active_futures['expiry_dt'] == nearest_expiry]
            else:
                print("DEBUG: No active futures found (all expired?)")
                df_fut = pd.DataFrame() # Force empty to trigger error

            # Drop temp column
            if not df_fut.empty:
                df_fut = df_fut.drop(columns=['expiry_dt'])
            
        futures_df = df_fut
        
    except Exception as e:
        st.error(f"Error loading NSE.json: {e}")
        # DEBUG: Print detailed error to console
        print(f"CRITICAL ERROR loading NSE.json: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()
        
    # Convert expiry to datetime
    if 'expiry' in futures_df.columns:
        futures_df['expiry_date'] = pd.to_datetime(futures_df['expiry'], unit='ms')
    if 'expiry' in options_df.columns:
        options_df['expiry_date'] = pd.to_datetime(options_df['expiry'], unit='ms')
    
    return futures_df, options_df

# --- Main App Logic ---

# Initialize data with spinner
with st.spinner("Initializing Application and Loading Data..."):
    futures_df, options_df = load_data()

if futures_df.empty or options_df.empty:
    st.error("Failed to load instruments data. Please check your internet connection and restart.")
    st.stop()

# --- API Functions ---
def get_ohlc(instrument_key, token):
    url = "https://api.upstox.com/v3/market-quote/ohlc"
    params = {
        'instrument_key': instrument_key,
        'interval': '1d'
    }
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'success':
            return data['data']
    except Exception as e:
        # st.error(f"Error fetching OHLC: {e}") # Suppress individual errors for batch
        pass
    return {}

def get_ltp(instrument_keys, token):
    url = "https://api.upstox.com/v3/market-quote/ltp"
    params = {'instrument_key': instrument_keys}
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'success':
            return data['data']
    except Exception as e:
        st.error(f"Error fetching LTP: {e}")
        print(f"LTP Fetch Error: {e}")
        pass
    return {}

# Helper function to find data by token
def find_data_by_token(token, data_dict):
    if not data_dict: return None
    # 1. Try direct key lookup
    if token in data_dict: return data_dict[token]
    if token.replace('|', ':') in data_dict: return data_dict[token.replace('|', ':')]
    
    # 2. Iterate and match instrument_token field
    for key, value in data_dict.items():
        if value.get('instrument_token') == token:
            return value
    return None

# --- Main Logic ---

# Auto-Refresh Controls
st.sidebar.markdown("---")
st.sidebar.header("Auto-Refresh Settings")
atm_mode = st.sidebar.radio("ATM Strike Based On:", ("Fixed (Open Price)", "Dynamic (LTP)"), index=0)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
refresh_interval = st.sidebar.number_input("Refresh Interval (seconds)", min_value=5, value=30, step=5)

# Determine if we should run
run_once = st.button("🔄 Refresh Data", type="primary")
should_run = run_once or auto_refresh

if should_run:
    # --- Time Restriction Check (09:00 AM - 03:40 PM IST) ---
    ist_now = get_ist_now()
    market_start = ist_now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = ist_now.replace(hour=15, minute=40, second=0, microsecond=0)
    
    # Check if current time is within trading hours
    is_market_closed = not (market_start <= ist_now <= market_end)
    
    if is_market_closed:
        st.warning(f"⚠️ Market Closed ({ist_now.strftime('%H:%M:%S')} IST). Auto-refresh is disabled. Showing final data.")
        # Proceed to fetch data once so the user can see the last state.
    
    if auto_refresh:
        st.caption(f"Auto-refreshing every {refresh_interval} seconds...")
        
    with st.spinner("Fetching and Calculating Data..."):
        # 1. Get unique futures (one per symbol)
        # Group by name and take the first one (assuming sorted by expiry/preference in CSV or just taking one)
        # We already filtered for current month in the CSV generation step.
        # To be safe, let's sort by expiry just in case
        futures_df_sorted = futures_df.sort_values('expiry_date')
        unique_futures = futures_df_sorted.drop_duplicates(subset=['name'], keep='first')
        
        all_results = []
        
        # Batch processing for Futures OHLC
        # Split unique_futures into chunks
        chunk_size = 20  # Reduced chunk size for better reliability
        future_records = unique_futures.to_dict('records')
        total_records = len(future_records)
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Helper to calculate percentage change
        def calc_pct_change(ltp, cp):
            if ltp is not None and cp and cp > 0:
                return ((ltp - cp) / cp) * 100
            return 0.0

        # We need to map Future Prices first
        future_prices = {} # {symbol_name: price}
        price_key = 'open' if atm_mode == "Fixed (Open Price)" else 'close'
        future_col_name = "Future Open" if atm_mode == "Fixed (Open Price)" else "Future LTP"
        
        # Prepare chunks
        chunks = [future_records[i:i+chunk_size] for i in range(0, total_records, chunk_size)]
        total_chunks = len(chunks)
        
        # Function to process a single chunk of futures
        def fetch_futures_chunk(chunk):
            keys = ",".join([r['instrument_key'] for r in chunk])
            ohlc_data = get_ohlc(keys, access_token)
            results = {}
            if ohlc_data:
                # Normalize keys
                lookup_map = {}
                for k, v in ohlc_data.items():
                    lookup_map[k] = v
                    if 'instrument_token' in v:
                        lookup_map[v['instrument_token']] = v
                
                for record in chunk:
                    key = record['instrument_key']
                    data = lookup_map.get(key)
                    if not data:
                        alt_key = key.replace('|', ':')
                        data = lookup_map.get(alt_key)
                    
                    if data:
                        ohlc_source = data.get('live_ohlc') or data.get('prev_ohlc')
                        if ohlc_source:
                            results[record['name']] = ohlc_source.get(price_key, 0.0)
            return results

        # Parallel Execution for Futures
        status_text.text(f"Fetching Futures Data... (0/{total_records})")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(fetch_futures_chunk, chunk): i for i, chunk in enumerate(chunks)}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    future_prices.update(chunk_results)
                except Exception as e:
                    # Log error but continue
                    print(f"Error fetching futures chunk: {e}")
                    pass
                
                completed_count += 1
                # Update Progress
                progress = min((completed_count / total_chunks) * 0.5, 0.5)
                progress_bar.progress(progress)
                status_text.text(f"Fetching Futures Data... ({completed_count * chunk_size}/{total_records})")

        # Prepare Option Keys to Fetch
        option_keys_to_fetch = []
        symbol_atm_map = {} # {symbol: {atm_strike, ce_key, pe_key}}
        
        # Optimization: Pre-filter and Group Options
        # 1. Filter only CE/PE
        # 2. Filter only symbols we have futures for
        relevant_options_df = options_df[
            (options_df['instrument_type'].isin(['CE', 'PE'])) &
            (options_df['name'].isin(future_prices.keys()))
        ]
        
        # Optimize Datetime Conversion ONCE (Vectorized)
        # Convert expiry to date object for faster comparison
        if not relevant_options_df.empty:
             relevant_options_df['expiry_date_obj'] = relevant_options_df['expiry_date'].dt.date
        
        # 3. Group by symbol for O(1) access
        options_grouped = relevant_options_df.groupby('name')
        
        for i, (symbol, ref_price) in enumerate(future_prices.items()):
            if ref_price <= 0: continue
            
            # Get options for this symbol from grouped dict
            if symbol not in options_grouped.groups:
                continue
                
            opts_group = options_grouped.get_group(symbol)
            
            # Get expiry from future record
            f_rec = unique_futures[unique_futures['name'] == symbol].iloc[0]
            f_expiry = f_rec['expiry_date'].date()
            
            # Filter by expiry (fast operation on small subset)
            # Use the pre-converted column
            opts = opts_group[opts_group['expiry_date_obj'] == f_expiry]
            
            if opts.empty: continue
            
            # Find nearest strike
            unique_strikes = sorted(opts['strike_price'].unique())
            if not unique_strikes: continue
            
            atm_strike = min(unique_strikes, key=lambda x: abs(x - ref_price))
            
            # Get CE and PE keys
            # Use boolean masking on small dataframe
            ce_row = opts[(opts['strike_price'] == atm_strike) & (opts['instrument_type'] == 'CE')]
            pe_row = opts[(opts['strike_price'] == atm_strike) & (opts['instrument_type'] == 'PE')]
            
            ce_key = ce_row.iloc[0]['instrument_key'] if not ce_row.empty else None
            pe_key = pe_row.iloc[0]['instrument_key'] if not pe_row.empty else None
            
            ce_lot = ce_row.iloc[0]['lot_size'] if not ce_row.empty else 0
            pe_lot = pe_row.iloc[0]['lot_size'] if not pe_row.empty else 0
            
            symbol_atm_map[symbol] = {
                'ref_price': ref_price,
                'atm_strike': atm_strike,
                'ce_key': ce_key,
                'pe_key': pe_key,
                'ce_lot': ce_lot,
                'pe_lot': pe_lot
            }
            
            if ce_key: option_keys_to_fetch.append(ce_key)
            if pe_key: option_keys_to_fetch.append(pe_key)
            
        # Batch Fetch Options Data
        options_data_map = {}
        
        total_opt_keys = len(option_keys_to_fetch)
        
        # Function to process options chunk
        def fetch_options_chunk(chunk_keys):
            # Upstox V3 API expects | in instrument_key (e.g. NSE_FO|12345)
            # Do NOT replace with : as that causes 400 Bad Request
            keys_str = ",".join(chunk_keys)
            return get_ltp(keys_str, access_token)

        if total_opt_keys > 0:
            opt_chunks = [option_keys_to_fetch[i:i+chunk_size] for i in range(0, total_opt_keys, chunk_size)]
            total_opt_chunks = len(opt_chunks)
            
            status_text.text(f"Fetching Options Data... (0/{total_opt_keys})")

            # DEBUG: Check first chunk keys and response
            # if opt_chunks:
            #     first_chunk = opt_chunks[0]
            #     # Show keys as they will be sent
            #     debug_keys_str = ",".join(first_chunk)
            #     st.write(f"DEBUG: First Chunk Keys (Request): {debug_keys_str}")
            #     try:
            #         debug_resp = get_ltp(debug_keys_str, access_token)
            #         st.write(f"DEBUG: First Chunk Response Keys: {list(debug_resp.keys()) if debug_resp else 'Empty Response'}")
            #     except Exception as ex:
            #         st.error(f"DEBUG: First Chunk Error: {ex}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_chunk = {executor.submit(fetch_options_chunk, chunk): i for i, chunk in enumerate(opt_chunks)}
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        ltp_data = future.result()
                        if ltp_data:
                            options_data_map.update(ltp_data)
                    except Exception as e:
                        # Log error but continue
                        print(f"Error fetching options chunk: {e}")
                        pass
                    
                    completed_count += 1
                    # Update Progress
                    progress = 0.5 + min((completed_count / total_opt_chunks) * 0.5, 0.5)
                    progress_bar.progress(progress)
                    status_text.text(f"Fetching Options Data... ({completed_count * chunk_size}/{total_opt_keys})")

        # Cleanup Progress
        progress_bar.progress(1.0)
        status_text.text("Finalizing Data...")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        if total_opt_keys > 0 and not options_data_map:
             st.error("Failed to fetch Options Data (LTP). Please check your Access Token or Internet Connection.")

        # Optimize Lookup for Options
        # Create a fast lookup map for options_data_map
        fast_options_map = {}
        for k, v in options_data_map.items():
            fast_options_map[k] = v
            
            # Map by Token from the VALUE object if available
            # This is the most robust way because the API might return keys in a different format
            if isinstance(v, dict) and 'instrument_token' in v:
                token = v['instrument_token']
                if token:
                    fast_options_map[str(token)] = v
            
            # Fallback: Map by Token (suffix) from the key
            # Key format is typically SEGMENT|TOKEN or SEGMENT:TOKEN
            try:
                token_from_key = k.replace(':', '|').split('|')[-1]
                fast_options_map[token_from_key] = v
            except:
                pass
            
        # DEBUG: Show sample of map if empty
        if not fast_options_map and total_opt_keys > 0:
             st.warning(f"Options Data Map is empty! Sent {total_opt_keys} keys.")

        # Construct Final DataFrame
        final_rows = []
        for symbol, info in symbol_atm_map.items():
            row = {
                "Stock Name": symbol,
                future_col_name: info['ref_price'],
                "ATM Strike": info['atm_strike']
            }
            
            # Helper to get data with fallback
            def get_opt_data(key):
                if not key: return None
                # Try exact match
                d = fast_options_map.get(key)
                if d: return d
                
                # Try token match (extract token from request key)
                try:
                    tok = key.replace(':', '|').split('|')[-1]
                    return fast_options_map.get(tok)
                except:
                    return None

            # CE Data
            ce_key = info['ce_key']
            ce_ltp = 0
            ce_pct = 0
            ce_vol = 0
            ce_ctr = 0
            
            if ce_key:
                # Optimized lookup
                data = get_opt_data(ce_key)
                if data:
                    ce_ltp = data.get('last_price', 0)
                    ce_vol = data.get('volume', 0)
                    ce_pct = calc_pct_change(ce_ltp, data.get('cp', 0))
                    # Calculate Contracts
                    lot_size = info.get('ce_lot', 0)
                    if lot_size > 0 and ce_vol > 0:
                         ce_ctr = ce_vol / lot_size
            
            row["CE LTP"] = ce_ltp
            row["CE Change %"] = round(ce_pct, 2)
            row["CE Volume"] = ce_vol
            row["CE Contracts"] = int(ce_ctr)
            
            # PE Data
            pe_key = info['pe_key']
            pe_ltp = 0
            pe_pct = 0
            pe_vol = 0
            pe_ctr = 0
            
            if pe_key:
                # Optimized lookup
                data = get_opt_data(pe_key)
                if data:
                    pe_ltp = data.get('last_price', 0)
                    pe_vol = data.get('volume', 0)
                    pe_pct = calc_pct_change(pe_ltp, data.get('cp', 0))
                    # Calculate Contracts
                    lot_size = info.get('pe_lot', 0)
                    if lot_size > 0 and pe_vol > 0:
                         pe_ctr = pe_vol / lot_size
                    
            row["PE LTP"] = pe_ltp
            row["PE Change %"] = round(pe_pct, 2)
            row["PE Volume"] = pe_vol
            row["PE Contracts"] = int(pe_ctr)
            
            final_rows.append(row)
            
        df_results = pd.DataFrame(final_rows)

        # Ensure ATM Strike is numeric and rounded for clean display
        if not df_results.empty and "ATM Strike" in df_results.columns:
            df_results["ATM Strike"] = df_results["ATM Strike"].astype(float).round(2)
        
        # Rename Stock Name to include timestamp for fullscreen visibility
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        df_results = df_results.rename(columns={"Stock Name": f"Stock Name ({time_str})"})
    
    # Styling Function
    def highlight_ce_pe(row):
        styles = []
        for col in row.index:
            if str(col).startswith("CE"):
                styles.append('background-color: #e6ffe6; color: black') # Light Green
            elif str(col).startswith("PE"):
                styles.append('background-color: #ffe6e6; color: black') # Light Red
            else:
                styles.append('')
        return styles

    # Display
    
    # Calculate height to show all rows (approx 35px per row + header)
    table_height = (len(df_results) + 1) * 35 + 3

    # Apply styling and explicit alignment
    styler = df_results.style.apply(highlight_ce_pe, axis=1)
    if "ATM Strike" in df_results.columns:
        styler = styler.set_properties(subset=['ATM Strike'], **{'text-align': 'right'})

    st.dataframe(
        styler,
        column_config={
            future_col_name: st.column_config.NumberColumn(format="%.2f"),
            "ATM Strike": st.column_config.NumberColumn(format="%g"),
            "CE LTP": st.column_config.NumberColumn(format="%.2f"),
            "PE LTP": st.column_config.NumberColumn(format="%.2f"),
            "CE Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "PE Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "CE Contracts": st.column_config.NumberColumn(format="%d"),
            "PE Contracts": st.column_config.NumberColumn(format="%d"),
        },
        use_container_width=True,
        hide_index=True,
        height=table_height
    )
    
    # Handle Auto-Refresh Loop
    if auto_refresh and not is_market_closed:
        time.sleep(refresh_interval)
        st.rerun()

elif not should_run:
    st.info("Click 'Load All Stocks Data' or enable 'Auto-Refresh' in the sidebar to start.")
