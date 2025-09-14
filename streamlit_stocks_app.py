#!/usr/bin/env python3
# streamlit_stocks_app.py
# Streamlit app to fetch and plot normalized stock prices using Plotly Express.
# Run:
#   pip install streamlit yfinance pandas plotly requests beautifulsoup4
#   streamlit run streamlit_stocks_app.py
#
import io
import time
from datetime import date, timedelta
from typing import List

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception as e:
    st.error("This app requires the 'yfinance' package. Install it with:  \n`pip install yfinance pandas streamlit plotly`")
    st.stop()

import plotly.express as px

try:
    import requests
    from bs4 import BeautifulSoup
except Exception as e:
    st.error("This app requires the 'requests' and 'beautifulsoup4' packages. Install them with:  \n`pip install requests beautifulsoup4`")
    st.stop()

# --------------
# Page settings
# --------------
st.set_page_config(page_title="Stock Plotter", layout="wide")
st.title("📈 Stonkatron")
st.caption("Compare tickers on the same scale over a chosen time range. Data via Yahoo Finance (yfinance).\
    \n Objective: Find Discounted Equities.")

# Tabs for main content
chart_tab, sp500_tab, pe_tab, movers_tab, growth_tab = st.tabs(["📊 Chart", "📋 S&P 500", "📐 P/E History", "📉 Weekly Movers", "💰 Growth"])

# --------------
# Sidebar controls
# --------------
with st.sidebar:
    st.header("Controls")
    tickers_input = st.text_input(
        "Tickers (comma or space separated)",
        value="SPY, AAPL, MSFT, NVDA, TSLA",
        help="Example: AAPL MSFT NVDA or AAPL,MSFT,NVDA",
    )

    today = date.today()
    range_options = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1yr": 365,
        "2yr": 730,
        "3yr": 365*3,
        "4yr": 365*4,
        "5yr": 1825,
        "10yr": 3650,
      "20yr": 365*20,
    }
    range_choice = st.radio("Select time range", options=list(range_options.keys()), index=3)
    days_back = range_options[range_choice]
    start_date = today - timedelta(days=days_back)
    end_date = today

    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    use_adjusted = st.checkbox("Use Adjusted Close (dividends/splits)", value=True)
    log_scale = st.checkbox("Log scale (y-axis)", value=False)

    norm_mode = st.radio(
        "Normalization baseline",
        options=["Each series starts at 1.0 on its first date", "Align to first common date then start at 1.0"],
        index=0,
        help=(
            "First option normalizes each ticker from its own first available date. "
            "Second option uses the intersection of dates across tickers, then normalizes."
        ),
    )

    # Auto-fetch when parameters change - no button needed
    submitted = True

    st.divider()
    st.header("Quick Yahoo Finance Lookup")
    lookup_ticker = st.text_input("Enter a single ticker", value="AAPL")
    if lookup_ticker.strip():
        yurl = f"https://finance.yahoo.com/quote/{lookup_ticker.strip().upper()}"
        st.markdown(f"[Open {lookup_ticker.strip().upper()} on Yahoo Finance]({yurl})")

# --------------
# Helper functions
# --------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, interval: str, use_adj: bool) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + pd.Timedelta(days=1),  # include end date
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    def extract_close(df: pd.DataFrame, tkr: str) -> pd.Series:
        if isinstance(df.columns, pd.MultiIndex):
            col = "Adj Close" if use_adj and (tkr, "Adj Close") in df.columns else "Close"
            return df[(tkr, col)].rename(tkr)
        else:
            col = "Adj Close" if use_adj and "Adj Close" in df.columns else "Close"
            return df[col].rename(tkr)

    if isinstance(data.columns, pd.MultiIndex):
        series = []
        for t in tickers:
            if (t, "Close") not in data.columns and (t, "Adj Close") not in data.columns:
                continue
            s = extract_close(data, t)
            if not s.dropna().empty:
                series.append(s)
        if not series:
            return pd.DataFrame()
        prices = pd.concat(series, axis=1)
    else:
        if data.empty:
            return pd.DataFrame()
        prices = extract_close(data, tickers[0]).to_frame()

    prices = prices.dropna(how="all")
    prices = prices.fillna(method="ffill").dropna(how="all")  # keep columns even if leading NaNs
    return prices

def normalize_df(prices: pd.DataFrame, align_common: bool) -> pd.DataFrame:
    df = prices.copy()
    if align_common:
        # Drop rows with any NaN so all tickers share the same index
        df = df.dropna(how="any")
    if df.empty:
        return df
    first_vals = df.iloc[0]
    return df.divide(first_vals)

def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

@st.cache_data(show_spinner=False)
def fetch_eps_ttm_quarterly(ticker: str) -> pd.Series:
    """Return a pandas Series of TTM EPS values indexed by report date for `ticker`.
    Tries multiple yfinance endpoints; falls back to a single trailing EPS if needed."""
    tk = yf.Ticker(ticker)
    eps_q = None
    try:
        inc = tk.get_income_stmt(freq="quarterly")
        if isinstance(inc, pd.DataFrame):
            for key in ["Diluted EPS", "DilutedEPS", "EPS Diluted", "Basic EPS", "BasicEPS"]:
                if key in inc.index:
                    eps_q = inc.loc[key]
                    break
    except Exception:
        pass
    if eps_q is None:
        try:
            inc = tk.quarterly_income_stmt
            if isinstance(inc, pd.DataFrame):
                for key in ["Diluted EPS", "DilutedEPS", "EPS Diluted", "Basic EPS", "BasicEPS"]:
                    if key in inc.index:
                        eps_row = inc.loc[key]
                        eps_q = pd.Series(eps_row.values, index=pd.to_datetime(eps_row.index))
                        break
        except Exception:
            pass
    if eps_q is None or (isinstance(eps_q, (pd.Series, pd.DataFrame)) and len(eps_q) == 0):
        teps = None
        try:
            fi = tk.fast_info
            teps = getattr(fi, "eps_ttm", None) if hasattr(fi, "eps_ttm") else fi.get("eps_ttm")
        except Exception:
            pass
        if teps is None:
            try:
                info = tk.info
                teps = info.get("trailingEps")
            except Exception:
                teps = None
        if teps is None or pd.isna(teps) or teps == 0:
            return pd.Series(dtype="float64")
        return pd.Series([float(teps)], index=[pd.Timestamp.today().normalize()])

    if isinstance(eps_q, pd.DataFrame):
        eps_q = eps_q.iloc[0]
    eps_q = pd.to_numeric(eps_q, errors="coerce").dropna()
    eps_q.index = pd.to_datetime(eps_q.index)
    eps_q = eps_q.sort_index()
    eps_ttm = eps_q.rolling(4).sum().dropna()
    return eps_ttm

@st.cache_data(show_spinner=False)
def fetch_eps_last_fy_series(ticker: str) -> pd.Series:
    """Return LAST-FISCAL-YEAR EPS values indexed by report date for `ticker`."""
    tk = yf.Ticker(ticker)
    eps_a = None
    try:
        inc = tk.get_income_stmt(freq="annual")
        if isinstance(inc, pd.DataFrame):
            for key in ["Diluted EPS", "DilutedEPS", "EPS Diluted", "Basic EPS", "BasicEPS"]:
                if key in inc.index:
                    eps_a = inc.loc[key]
                    break
    except Exception:
        pass
    if eps_a is None:
        try:
            inc = tk.income_stmt
            if isinstance(inc, pd.DataFrame):
                for key in ["Diluted EPS", "DilutedEPS", "EPS Diluted", "Basic EPS", "BasicEPS"]:
                    if key in inc.index:
                        eps_row = inc.loc[key]
                        eps_a = pd.Series(eps_row.values, index=pd.to_datetime(eps_row.index))
                        break
        except Exception:
            pass
    if eps_a is None:
        return pd.Series(dtype="float64")
    if isinstance(eps_a, pd.DataFrame):
        eps_a = eps_a.iloc[0]
    eps_a = pd.to_numeric(eps_a, errors="coerce").dropna()
    eps_a.index = pd.to_datetime(eps_a.index)
    eps_a = eps_a.sort_index()
    return eps_a

@st.cache_data(show_spinner=False)
def fetch_macrotrends_pe(ticker: str) -> pd.DataFrame:
    """Fetch P/E ratio data from Macrotrends for a given ticker with rate limiting."""
    try:
        url = f"https://macrotrends.net/stocks/charts/{ticker.upper()}/{ticker.lower()}/pe-ratio"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        
        # Add delay to avoid rate limiting
        time.sleep(1.5)
        
        response = requests.get(url, headers=headers, timeout=15)
        
        # Handle rate limiting specifically
        if response.status_code == 429:
            st.warning(f"Rate limited for {ticker}. Please wait a moment and try again with fewer tickers.")
            return pd.DataFrame()
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the first table (historical data table)
        tables = soup.find_all('table')
        if not tables:
            return pd.DataFrame()
        
        # The first table contains the historical P/E data
        table = tables[0]
        rows = table.find_all('tr')
        
        # Skip the first row (title) and second row (headers)
        data_rows = rows[2:] if len(rows) > 2 else []
        data = []
        
        for row in data_rows:
            cells = row.find_all('td')
            if len(cells) >= 4:  # Date, Stock Price, TTM Net EPS, PE Ratio
                try:
                    date_str = cells[0].get_text(strip=True)
                    stock_price = cells[1].get_text(strip=True).replace('$', '').replace(',', '')
                    eps = cells[2].get_text(strip=True).replace('$', '').replace(',', '')
                    pe_ratio = cells[3].get_text(strip=True).replace(',', '')
                    
                    # Skip rows with missing or invalid data
                    if not date_str or not pe_ratio or pe_ratio == '':
                        continue
                    
                    # Convert to appropriate types
                    stock_price = float(stock_price) if stock_price != '' else None
                    eps = float(eps) if eps != '' else None
                    pe_ratio = float(pe_ratio) if pe_ratio != '' and pe_ratio != '0.00' else None
                    
                    if pe_ratio and pe_ratio > 0:  # Only require valid P/E ratio
                        data.append({
                            'Date': pd.to_datetime(date_str),
                            'Stock_Price': stock_price,
                            'EPS': eps,
                            'PE_Ratio': pe_ratio
                        })
                except (ValueError, TypeError) as e:
                    continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        return df[['PE_Ratio']]  # Return only P/E ratios
        
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            st.warning(f"Rate limited for {ticker}. Please wait a moment and try again with fewer tickers.")
        else:
            st.warning(f"Network error for {ticker}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Failed to fetch Macrotrends data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_current_pe_ratios(tickers: List[str]) -> pd.DataFrame:
    """Fetch current P/E ratios for given tickers."""
    result = {}
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            # Try multiple P/E ratio fields
            pe_ratio = None
            for key in ['trailingPE', 'forwardPE', 'priceToEarnings']:
                if key in info and info[key] is not None:
                    pe_ratio = info[key]
                    break
            
            if pe_ratio is not None and not pd.isna(pe_ratio) and pe_ratio > 0:
                result[ticker] = pe_ratio
                
        except Exception:
            continue
    
    if not result:
        return pd.DataFrame()
    
    return pd.DataFrame(list(result.items()), columns=['Ticker', 'PE_Ratio']).set_index('Ticker')

@st.cache_data(show_spinner=False)
def fetch_current_ev_ebitda_ratios(tickers: List[str]) -> pd.DataFrame:
    """Fetch current EV/EBITDA ratios for given tickers."""
    result = {}
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            # Get enterprise value and EBITDA
            enterprise_value = info.get('enterpriseValue', None)
            ebitda = info.get('ebitda', None)
            
            # Calculate EV/EBITDA ratio
            if (enterprise_value is not None and not pd.isna(enterprise_value) and enterprise_value > 0 and
                ebitda is not None and not pd.isna(ebitda) and ebitda > 0):
                ev_ebitda_ratio = enterprise_value / ebitda
                result[ticker] = ev_ebitda_ratio
                
        except Exception:
            continue
    
    if not result:
        return pd.DataFrame()
    
    return pd.DataFrame(list(result.items()), columns=['Ticker', 'EV_EBITDA_Ratio']).set_index('Ticker')

@st.cache_data(show_spinner=False)
def fetch_ev_ebitda_data(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch EV/EBITDA data for given tickers."""
    result = {}
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            
            # Get current market cap and enterprise value
            info = tk.info
            market_cap = info.get('marketCap', None)
            enterprise_value = info.get('enterpriseValue', None)
            
            # Get quarterly financial data
            financials = tk.quarterly_financials
            if financials is None or financials.empty:
                continue
                
            # Look for EBITDA in various possible keys
            ebitda_key = None
            for key in ['EBITDA', 'EBIT', 'Operating Income', 'OperatingIncome']:
                if key in financials.index:
                    ebitda_key = key
                    break
            
            if ebitda_key is None:
                continue
                
            ebitda_series = financials.loc[ebitda_key]
            ebitda_series = pd.to_numeric(ebitda_series, errors='coerce').dropna()
            ebitda_series.index = pd.to_datetime(ebitda_series.index)
            
            # Calculate EV/EBITDA ratio
            if enterprise_value and not pd.isna(enterprise_value) and enterprise_value > 0:
                # Use enterprise value if available
                ev_ebitda = enterprise_value / ebitda_series
            elif market_cap and not pd.isna(market_cap) and market_cap > 0:
                # Fall back to market cap if enterprise value not available
                ev_ebitda = market_cap / ebitda_series
            else:
                continue
                
            # Filter by date range
            ev_ebitda = ev_ebitda.loc[(ev_ebitda.index >= start) & (ev_ebitda.index <= end)]
            ev_ebitda = ev_ebitda.dropna()
            
            if not ev_ebitda.empty:
                result[ticker] = ev_ebitda
                
        except Exception as e:
            continue
    
    if not result:
        return pd.DataFrame()
    
    # Combine all series
    ev_ebitda_df = pd.DataFrame(result)
    return ev_ebitda_df

@st.cache_data(show_spinner=False)
def calculate_compound_growth(starting_value: float, annual_growth_rate: float, years: int, annual_investment: float = 0) -> pd.DataFrame:
    """Calculate compound growth over time with optional annual contributions."""
    data = []
    current_value = starting_value
    total_contributions = starting_value
    
    for year in range(years + 1):
        # Calculate gains from growth
        gains_from_growth = current_value - total_contributions
        
        data.append({
            'Year': year,
            'Value': current_value,
            'Total_Contributions': total_contributions,
            'Gains_From_Growth': gains_from_growth,
            'Yearly_Contribution': annual_investment if year > 0 else 0,
            'Growth_Rate': annual_growth_rate
        })
        
        if year < years:  # Don't compound after the last year
            # Add annual contribution at the beginning of the year
            if annual_investment > 0:
                current_value += annual_investment
                total_contributions += annual_investment
            
            # Apply growth for the year
            current_value = current_value * (1 + annual_growth_rate / 100)
    
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def build_pe_timeseries(prices: pd.DataFrame, basis: str) -> pd.DataFrame:
    result = {}
    for t in prices.columns:
        try:
            if basis == "Last FY":
                eps_series = fetch_eps_last_fy_series(t)
            else:
                eps_series = fetch_eps_ttm_quarterly(t)
            if eps_series.empty:
                continue
            eps_aligned = eps_series.reindex(prices.index, method="ffill")
            pe = prices[t] / eps_aligned
            result[t] = pe
        except Exception:
            continue
    if not result:
        return pd.DataFrame()
    pe_df = pd.DataFrame(result, index=prices.index)
    return pe_df

# --------------
# Main logic
# --------------
# Chart tab
# --------------
with chart_tab:
    # Parse tickers
    raw = tickers_input.replace(",", " ").upper().split()
    tickers = sorted(set([t.strip() for t in raw if t.strip()]))

    if not tickers:
        st.warning("Please enter at least one ticker.")
    elif start_date >= end_date:
        st.warning("Start date must be before end date.")
    else:
        with st.spinner("Fetching data..."):
            prices = fetch_prices(tickers, pd.Timestamp(start_date), pd.Timestamp(end_date), interval, use_adjusted)

        if prices.empty:
            st.error("No price data found for the given inputs. Try different dates/tickers.")
        else:
            # Normalize
            align_common = (norm_mode.startswith("Align"))
            norm = normalize_df(prices, align_common=align_common)

            if norm.empty:
                st.error("Normalization produced an empty dataset (likely no overlapping dates). Try the other baseline option.")
            else:
                # Plot with Plotly Express
                norm_reset = norm.reset_index()
                norm_reset = norm_reset.rename(columns={norm_reset.columns[0]: "Date"})
                norm_reset = norm_reset.melt(id_vars=["Date"], var_name="Ticker", value_name="Normalized")
                fig = px.line(
                    norm_reset,
                    x="Date",
                    y="Normalized",
                    color="Ticker",
                    title=f"Normalized Price",
                    log_y=log_scale,
                    width=1500,
                    height=900,
                )
                fig.update_layout(legend_title_text="Ticker")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add current ratios bar charts
                st.subheader("📊 Current Valuation Ratios")
                st.caption("Latest P/E and EV/EBITDA ratios for quick comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current P/E Ratios")
                    with st.spinner("Fetching current P/E ratios..."):
                        current_pe_df = fetch_current_pe_ratios(tickers)
                    
                    if current_pe_df.empty:
                        st.warning("No current P/E data available for the selected tickers.")
                    else:
                        # Sort by P/E ratio (ascending - lower is better)
                        current_pe_sorted = current_pe_df.sort_values('PE_Ratio')
                        
                        fig_pe_bar = px.bar(
                            current_pe_sorted.reset_index(),
                            x='Ticker',
                            y='PE_Ratio',
                            title="Current P/E Ratios",
                            color='PE_Ratio',
                            color_continuous_scale='RdYlGn_r',  # Red for high, Green for low
                            height=400
                        )
                        fig_pe_bar.update_layout(
                            xaxis_title="Ticker",
                            yaxis_title="P/E Ratio",
                            showlegend=False
                        )
                        st.plotly_chart(fig_pe_bar, use_container_width=True)
                        
                        # Show data table
                        # st.dataframe(current_pe_sorted, use_container_width=True)
                
                with col2:
                    st.subheader("Current EV/EBITDA Ratios")
                    with st.spinner("Fetching current EV/EBITDA ratios..."):
                        current_ev_df = fetch_current_ev_ebitda_ratios(tickers)
                    
                    if current_ev_df.empty:
                        st.warning("No current EV/EBITDA data available for the selected tickers.")
                    else:
                        # Sort by EV/EBITDA ratio (ascending - lower is better)
                        current_ev_sorted = current_ev_df.sort_values('EV_EBITDA_Ratio')
                        
                        fig_ev_bar = px.bar(
                            current_ev_sorted.reset_index(),
                            x='Ticker',
                            y='EV_EBITDA_Ratio',
                            title="Current EV/EBITDA Ratios",
                            color='EV_EBITDA_Ratio',
                            color_continuous_scale='RdYlGn_r',  # Red for high, Green for low
                            height=400
                        )
                        fig_ev_bar.update_layout(
                            xaxis_title="Ticker",
                            yaxis_title="EV/EBITDA Ratio",
                            showlegend=False
                        )
                        st.plotly_chart(fig_ev_bar, use_container_width=True)
                        
                        # Show data table
                        st.dataframe(current_ev_sorted, use_container_width=True)
                
                st.info("💡 **Lower ratios** in both charts may indicate better value opportunities. Compare within the same industry for context.")
                
                # Add EV/EBITDA chart below the normalized price chart
                st.subheader("📊 EV/EBITDA Ratio")
                st.caption("Enterprise Value to EBITDA ratio - lower values may indicate undervaluation")
                
                with st.spinner("Fetching EV/EBITDA data..."):
                    ev_ebitda_df = fetch_ev_ebitda_data(tickers, pd.Timestamp(start_date), pd.Timestamp(end_date))
                
                if ev_ebitda_df.empty:
                    st.warning("No EV/EBITDA data available for the selected tickers. This data may not be available for all companies.")
                else:
                    # Plot EV/EBITDA data
                    ev_ebitda_tidy = ev_ebitda_df.reset_index().melt(id_vars=[ev_ebitda_df.index.name or "index"], var_name="Ticker", value_name="EV_EBITDA")
                    ev_ebitda_tidy = ev_ebitda_tidy.rename(columns={ev_ebitda_tidy.columns[0]: "Date"})
                    ev_ebitda_tidy = ev_ebitda_tidy.dropna(subset=['EV_EBITDA'])
                    
                    if not ev_ebitda_tidy.empty:
                        fig_ev = px.line(
                            ev_ebitda_tidy,
                            x="Date",
                            y="EV_EBITDA",
                            color="Ticker",
                            title="EV/EBITDA Ratio Over Time",
                            width=1500,
                            height=600,
                        )
                        fig_ev.update_layout(legend_title_text="Ticker")
                        st.plotly_chart(fig_ev, use_container_width=True)
                        
                        # Show latest EV/EBITDA values
                        ev_ebitda_clean = ev_ebitda_df.dropna()
                        if not ev_ebitda_clean.empty:
                            latest_ev = ev_ebitda_clean.iloc[-1].sort_values(ascending=True)
                            latest_ev_df = latest_ev.to_frame(name="Latest EV/EBITDA")
                            st.dataframe(latest_ev_df, use_container_width=True)
                        else:
                            st.warning("No valid EV/EBITDA data points available for display.")
                        
                        st.info("💡 **Lower EV/EBITDA ratios** may indicate better value opportunities, but consider industry context and company fundamentals.")
                    else:
                        st.warning("No valid EV/EBITDA data points found after filtering.")


# -----------------------------
# P/E tab
# -----------------------------
with pe_tab:
    st.subheader("P/E Ratio History")
    st.caption("Choose data source: **Yahoo Finance** (computed from Price/EPS) or **Macrotrends** (historical P/E tables).")
    
    # Data source selection
    data_source = st.radio("Data Source", ["Yahoo Finance", "Macrotrends"], horizontal=True, index=0)
    
    if data_source == "Yahoo Finance":
        st.caption("Computed as Price / EPS. Choose EPS basis: **TTM** (trailing twelve months) or **Last FY** (most recent fiscal year).")
        eps_basis = st.radio("EPS basis", ["TTM", "Last FY"], horizontal=True, index=0)

        # Parse tickers for P/E analysis
        raw_pe = tickers_input.replace(",", " ").upper().split()
        tickers_pe = sorted(set([s.strip() for s in raw_pe if s.strip()]))
        
        if not tickers_pe:
            st.warning("Please enter at least one ticker in the sidebar.")
        elif start_date >= end_date:
            st.warning("Start date must be before end date.")
        else:
            with st.spinner("Computing P/E histories..."):
                prices_pe = fetch_prices(tickers_pe, pd.Timestamp(start_date), pd.Timestamp(end_date), interval, use_adjusted)
            
            if prices_pe.empty:
                st.warning("No price data available for the selected tickers/date range.")
            else:
                pe_df = build_pe_timeseries(prices_pe, eps_basis)
                if pe_df.empty:
                    st.warning("Could not compute P/E series (missing EPS data). Try different tickers or a longer range.")
                else:
                    pe_tidy = pe_df.reset_index().melt(id_vars=[pe_df.index.name or "index"], var_name="Ticker", value_name="PE")
                    pe_tidy = pe_tidy.rename(columns={pe_tidy.columns[0]: "Date"})
                    fig_pe = px.line(
                        pe_tidy,
                        x="Date",
                        y="PE",
                        color="Ticker",
                        title=f"P/E ({eps_basis}) over time",
                        width=1500,
                        height=900,
                    )
                    fig_pe.update_layout(legend_title_text="Ticker")
                    st.plotly_chart(fig_pe, use_container_width=True)

                    pe_clean = pe_df.dropna()
                    if not pe_clean.empty:
                        latest = pe_clean.iloc[-1].sort_values(ascending=False)
                        latest_df = latest.to_frame(name="Latest P/E")
                        st.dataframe(latest_df, use_container_width=True)
                    else:
                        st.warning("No valid P/E data points available for display.")
    
    else:  # Macrotrends
        st.caption("Fetch historical P/E data directly from Macrotrends tables. Uses tickers and time range from sidebar controls.")
        st.info("💡 **Tip**: Macrotrends has rate limits. For best results, use 3-10 tickers at a time. The app will automatically limit to 10 tickers if you enter more.")
        
        # Parse tickers for Macrotrends analysis (use sidebar tickers)
        raw_mt = tickers_input.replace(",", " ").upper().split()
        tickers_mt = sorted(set([s.strip() for s in raw_mt if s.strip()]))
        
        if not tickers_mt:
            st.warning("Please enter at least one ticker in the sidebar.")
        elif start_date >= end_date:
            st.warning("Start date must be before end date.")
        else:
            # Limit number of tickers to avoid rate limiting
            if len(tickers_mt) > 10 :
                st.warning(f"⚠️ Too many tickers ({len(tickers_mt)}). Macrotrends has rate limits. Showing first 10 tickers only.")
                tickers_mt = tickers_mt[:10]
            
            with st.spinner("Fetching P/E data from Macrotrends (this may take a while due to rate limiting)..."):
                pe_data_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, ticker in enumerate(tickers_mt):
                    status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers_mt)})")
                    pe_data = fetch_macrotrends_pe(ticker)
                    if not pe_data.empty:
                        pe_data.columns = [ticker]  # Rename column to ticker
                        pe_data_list.append(pe_data)
                        st.success(f"✅ Successfully fetched {ticker}")
                    else:
                        st.warning(f"⚠️ No data found for {ticker}")
                    
                    progress_bar.progress((i + 1) / len(tickers_mt))
                
                status_text.text("Complete!")
                progress_bar.empty()
                status_text.empty()
                
                if not pe_data_list:
                    st.warning("No P/E data found from Macrotrends for the given tickers.")
                else:
                    # Combine all P/E data
                    pe_df_mt = pd.concat(pe_data_list, axis=1)
                    pe_df_mt = pe_df_mt.dropna(how='all')  # Remove rows where all values are NaN
                    
                    if pe_df_mt.empty:
                        st.warning("No overlapping P/E data found for the selected tickers.")
                    else:
                        # Filter data based on sidebar time range
                        pe_df_mt_filtered = pe_df_mt.loc[
                            (pe_df_mt.index >= pd.Timestamp(start_date)) & 
                            (pe_df_mt.index <= pd.Timestamp(end_date))
                        ]
                        
                        if pe_df_mt_filtered.empty:
                            st.warning(f"No P/E data found in the selected time range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).")
                        else:
                            # Plot Macrotrends P/E data
                            pe_tidy_mt = pe_df_mt_filtered.reset_index().melt(id_vars=[pe_df_mt_filtered.index.name or "index"], var_name="Ticker", value_name="PE")
                            pe_tidy_mt = pe_tidy_mt.rename(columns={pe_tidy_mt.columns[0]: "Date"})
                            pe_tidy_mt = pe_tidy_mt.dropna(subset=['PE'])  # Remove NaN values
                            
                            if not pe_tidy_mt.empty:
                                fig_pe_mt = px.line(
                                    pe_tidy_mt,
                                    x="Date",
                                    y="PE",
                                    color="Ticker",
                                    title=f"P/E Ratio History (Macrotrends) - {range_choice}",
                                    width=1500,
                                    height=900,
                                )
                                fig_pe_mt.update_layout(legend_title_text="Ticker")
                                st.plotly_chart(fig_pe_mt, use_container_width=True)

                                # Show latest P/E values
                                pe_mt_clean = pe_df_mt_filtered.dropna()
                                if not pe_mt_clean.empty:
                                    latest_mt = pe_mt_clean.iloc[-1].sort_values(ascending=False)
                                    latest_df_mt = latest_mt.to_frame(name="Latest P/E (Macrotrends)")
                                    st.dataframe(latest_df_mt, use_container_width=True)
                                else:
                                    st.warning("No valid P/E data points available for display.")
                                
                                # Show data range info
                                st.info(f"Showing data from {pe_df_mt_filtered.index.min().strftime('%Y-%m-%d')} to {pe_df_mt_filtered.index.max().strftime('%Y-%m-%d')} (filtered by sidebar time range)")
                            else:
                                st.warning("No valid P/E data points found after filtering.")

# -----------------------------
# S&P 500 tab
# -----------------------------
with sp500_tab:
    st.subheader("S&P 500 Constituents (from Wikipedia)")

    @st.cache_data(show_spinner=False)
    def load_sp500() -> pd.DataFrame:
        """Load S&P 500 table from Wikipedia with a browser-like User-Agent to avoid 403s."""
        import requests
        from io import StringIO

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()  # will raise HTTPError for 4xx/5xx

        # Parse the first table that contains the constituents
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            raise ValueError("No tables found on Wikipedia page.")
        df = tables[0].copy()

        # Normalize tickers for Yahoo Finance (e.g., BRK.B -> BRK-B)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.replace(r"\\.", "-", regex=True)
        return df

    sp500_df = load_sp500()
    st.caption("Symbols use '-' instead of '.' to match Yahoo Finance ticker format.")
    st.dataframe(sp500_df, use_container_width=True)

    # Convenience: export and copy tickers
    tickers_txt = " ".join(sp500_df.get("Symbol", pd.Series(dtype=str)).tolist())
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.download_button(
            "Download S&P 500 CSV",
            data=sp500_df.to_csv(index=False).encode("utf-8"),
            file_name="sp500_constituents.csv",
            mime="text/csv",
        )
    with col_b:
        st.text_area("All tickers (space-separated)", value=tickers_txt, height=120)

    st.divider()
    st.subheader("Rank by Market Cap")

    @st.cache_data(show_spinner=False)
    def fetch_market_caps(symbols: list[str]) -> pd.DataFrame:
        rows = []
        for t in symbols:
            try:
                fi = yf.Ticker(t).fast_info
                mc = getattr(fi, "market_cap", None) if hasattr(fi, "market_cap") else fi.get("market_cap")
                if mc is None:
                    info = yf.Ticker(t).info
                    mc = info.get("marketCap") if isinstance(info, dict) else None
                rows.append({"Symbol": t, "MarketCap": mc})
            except Exception:
                rows.append({"Symbol": t, "MarketCap": None})
        df = pd.DataFrame(rows)
        df = df.groupby("Symbol", as_index=False)["MarketCap"].max()
        return df

    def humanize_cap(x: float) -> str:
        try:
            x = float(x)
        except Exception:
            return ""
        if x >= 1e12:
            return f"{x/1e12:.2f}T"
        if x >= 1e9:
            return f"{x/1e9:.2f}B"
        if x >= 1e6:
            return f"{x/1e6:.2f}M"
        return f"{x:,.0f}"

    top_n = st.slider("Show top N by market cap", min_value=1, max_value=100, value=10, step=1)
    if st.button("Fetch & Rank"):
        with st.spinner("Fetching market caps from Yahoo Finance… this can take ~10–30s"):
            caps = fetch_market_caps(sp500_df["Symbol"].tolist())
        ranked = sp500_df.merge(caps, on="Symbol", how="left").sort_values("MarketCap", ascending=False)
        ranked["MarketCap (human)"] = ranked["MarketCap"].apply(humanize_cap)

        st.dataframe(ranked.head(top_n), use_container_width=True)
        st.download_button(
            "Download ranked CSV",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="sp500_ranked_by_market_cap.csv",
            mime="text/csv",
        )

        topN = ranked.head(top_n)
        ranked_txt = " ".join(topN["Symbol"].dropna().tolist())
        st.text_area("Top N tickers (space-separated)", value=ranked_txt, height=120)

    st.divider()
    st.subheader("Tickers by GICS Sub-Industry")
    if "GICS Sub-Industry" in sp500_df.columns:
        subindustries = sorted(sp500_df["GICS Sub-Industry"].dropna().unique())
        choice = st.selectbox("Select a GICS Sub-Industry", options=subindustries)
        filtered = sp500_df[sp500_df["GICS Sub-Industry"] == choice]
        st.dataframe(filtered, use_container_width=True)
        subs_txt = " ".join(filtered["Symbol"].dropna().tolist())
        st.text_area("Tickers in this sub-industry", value=subs_txt, height=120)

# -----------------------------
# Movers tab
# -----------------------------
with movers_tab:
    st.subheader("Weekly Movers (S&P 500)")
    st.caption(
        "Find the biggest losers or most volatile S&P 500 stocks over a selectable window. "
        "Volatility is the standard deviation of daily returns over the chosen trading-day window."
    )

    # Modular timeframe selector
    timeframe_options = {
        "1wk (5d)": 5,
        "2wks (10d)": 10,
        "1mo (21d)": 21,
        "2mo (42d)": 42,
        "6mo (126d)": 126,
        "1yr (252d)": 252,
    }
    col_tf1, col_tf2 = st.columns([1, 1])
    with col_tf1:
        metric = st.radio(
            "Metric",
            ["Biggest Losers", "Most Volatile"],
            horizontal=True,
            index=0,
        )
    with col_tf2:
        tf_label = st.selectbox("Window", options=list(timeframe_options.keys()), index=0)
    ndays = timeframe_options[tf_label]

    top_n_movers = st.slider("Show top N", min_value=1, max_value=50, value=5, step=1)

    if st.button("Compute Movers"):
        try:
            sp500_all = load_sp500()
            symbols = sp500_all.get("Symbol", pd.Series(dtype=str)).dropna().tolist()

            # Pull enough calendar days to comfortably cover the requested number of trading days
            end_dt = pd.Timestamp(date.today())
            calendar_buffer_days = int(ndays * 2) + 10  # generous buffer for weekends/holidays
            start_dt = end_dt - pd.Timedelta(days=calendar_buffer_days)

            with st.spinner(f"Fetching prices for S&P 500 (covering ~{ndays} trading days)…"):
                px_sp = fetch_prices(symbols, start_dt, end_dt, "1d", use_adjusted)

            if px_sp.empty:
                st.warning("No recent price data returned.")
            else:
                px_sp = px_sp.sort_index()

                # Generic N-day return: last price vs price N trading days ago
                def nday_return(s: pd.Series, n: int):
                    s = s.dropna()
                    if len(s) <= n:
                        return pd.NA
                    return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1)

                nret = px_sp.apply(nday_return, n=ndays)
                rets = px_sp.pct_change()
                vol_nd = rets.tail(ndays).std()

                movers_df = pd.DataFrame({
                    "Symbol": nret.index,
                    "Return_N": nret.values,
                    "Volatility_N": vol_nd.reindex(nret.index).values,
                }).dropna(subset=["Return_N"])  # require valid return for ranking

                if metric == "Biggest Losers":
                    view = movers_df.sort_values("Return_N", ascending=True).head(top_n_movers)
                    ycol = "Return_N"
                    title = f"S&P 500 — Biggest Losers over {tf_label}"
                else:
                    view = movers_df.dropna(subset=["Volatility_N"]).sort_values("Volatility_N", ascending=False).head(top_n_movers)
                    ycol = "Volatility_N"
                    title = f"S&P 500 — Most Volatile over {tf_label}"

                # Pretty percent for return column when present
                def to_pct(x):
                    try:
                        return f"{x*100:.2f}%"
                    except Exception:
                        return ""

                view_disp = view.copy()
                if "Return_N" in view_disp.columns:
                    view_disp["Return_N"] = view_disp["Return_N"].apply(to_pct)

                st.dataframe(view_disp, use_container_width=True)

                import plotly.express as px
                fig_mv = px.bar(view, x="Symbol", y=ycol, title=title, width=1500, height=700)
                st.plotly_chart(fig_mv, use_container_width=True)

                tick_txt = " ".join(view["Symbol"].tolist())
                st.text_area("Tickers in this list (space-separated)", value=tick_txt, height=100)
        except Exception as ex:
            st.error(f"Failed to compute movers: {ex}")

# -----------------------------
# Growth Calculator tab
# -----------------------------
with growth_tab:
    st.subheader("💰 Compound Growth Calculator")
    st.caption("Calculate the future value of investments with compound growth")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        starting_value = st.number_input(
            "Starting Value ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Initial investment amount"
        )
    
    with col2:
        annual_growth_rate = st.number_input(
            "Annual Growth Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=7.0,
            step=0.1,
            help="Expected annual return percentage"
        )
    
    with col3:
        annual_investment = st.number_input(
            "Annual Investment ($)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            help="Additional amount invested each year"
        )
    
    with col4:
        years = st.number_input(
            "Number of Years",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="Investment time horizon"
        )
    
    # Calculate compound growth
    if starting_value > 0 and annual_growth_rate >= 0 and years > 0:
        growth_df = calculate_compound_growth(starting_value, annual_growth_rate, years, annual_investment)
        
        # Display key metrics
        final_value = growth_df.iloc[-1]['Value']
        total_contributions = growth_df.iloc[-1]['Total_Contributions']
        total_gains = growth_df.iloc[-1]['Gains_From_Growth']
        total_return_pct = (total_gains / total_contributions) * 100 if total_contributions > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Final Value", f"${final_value:,.2f}")
        with col2:
            st.metric("Total Contributions", f"${total_contributions:,.2f}")
        with col3:
            st.metric("Total Gains", f"${total_gains:,.2f}")
        with col4:
            st.metric("Return %", f"{total_return_pct:.1f}%")
        with col5:
            if annual_investment > 0:
                st.metric("Annual Investment", f"${annual_investment:,.0f}")
            else:
                st.metric("Annual Investment", "None")
        
        # Create growth chart with contributions and gains
        fig_growth = px.line(
            growth_df,
            x='Year',
            y=['Total_Contributions', 'Value'],
            title=f"Investment Growth: ${starting_value:,.0f} + ${annual_investment:,.0f}/year at {annual_growth_rate}% for {years} years",
            height=500,
            color_discrete_map={
                'Total_Contributions': '#1f77b4',
                'Value': '#ff7f0e'
            }
        )
        fig_growth.update_layout(
            xaxis_title="Years",
            yaxis_title="Value ($)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig_growth.update_traces(
            line=dict(width=3),
            hovertemplate="Year %{x}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>"
        )
        # Rename legend labels
        fig_growth.for_each_trace(lambda t: t.update(name="Total Value" if t.name == "Value" else "Total Contributions"))
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # Add comparison chart if annual investment > 0
        if annual_investment > 0:
            # Calculate scenario without annual investments for comparison
            no_annual_df = calculate_compound_growth(starting_value, annual_growth_rate, years, 0)
            
            comparison_df = pd.DataFrame({
                'Year': growth_df['Year'],
                'With Annual Investment': growth_df['Value'],
                'Lump Sum Only': no_annual_df['Value']
            })
            
            fig_comparison = px.line(
                comparison_df,
                x='Year',
                y=['With Annual Investment', 'Lump Sum Only'],
                title="Impact of Annual Contributions vs Lump Sum Only",
                height=400,
                color_discrete_map={
                    'With Annual Investment': '#2ca02c',
                    'Lump Sum Only': '#d62728'
                }
            )
            fig_comparison.update_layout(
                xaxis_title="Years",
                yaxis_title="Value ($)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_comparison.update_traces(
                line=dict(width=3),
                hovertemplate="Year %{x}<br>%{fullData.name}: $%{y:,.2f}<extra></extra>"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Show the difference
            final_difference = final_value - no_annual_df.iloc[-1]['Value']
            st.success(f"🎯 **Annual contributions add ${final_difference:,.2f}** to your final value compared to lump sum only!")
        
        # Show detailed table
        st.subheader("📊 Year-by-Year Breakdown")
        
        # Format the dataframe for display
        display_df = growth_df.copy()
        display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Total_Contributions'] = display_df['Total_Contributions'].apply(lambda x: f"${x:,.2f}")
        display_df['Gains_From_Growth'] = display_df['Gains_From_Growth'].apply(lambda x: f"${x:,.2f}")
        display_df['Yearly_Contribution'] = display_df['Yearly_Contribution'].apply(lambda x: f"${x:,.2f}")
        display_df['Growth_Rate'] = display_df['Growth_Rate'].apply(lambda x: f"{x:.1f}%")
        display_df = display_df.rename(columns={
            'Value': 'Total Value',
            'Total_Contributions': 'Total Contributions',
            'Gains_From_Growth': 'Gains from Growth',
            'Yearly_Contribution': 'Yearly Contribution',
            'Growth_Rate': 'Annual Rate'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Additional insights
        st.subheader("💡 Key Insights")
        
        # Calculate doubling time (rule of 72)
        if annual_growth_rate > 0:
            doubling_time = 72 / annual_growth_rate
            st.info(f"**Rule of 72**: Your investment will approximately double every {doubling_time:.1f} years at {annual_growth_rate}% annual growth.")
        
        # Show year when value reaches certain milestones
        milestones = [starting_value * 2, starting_value * 5, starting_value * 10]
        milestone_names = ["2x", "5x", "10x"]
        
        # for milestone, name in zip(milestones, milestone_names):
            # if final_value >= milestone:
                # milestone_year = growth_df[growth_df['Value'] >= milestone]['Year'].iloc[0]
                # st.success(f"🎯 **{name} milestone** reached in year {milestone_year}")
        
        # Download data
        csv_data = growth_df.to_csv(index=False)
        st.download_button(
            "📥 Download Growth Data (CSV)",
            data=csv_data,
            file_name=f"compound_growth_{starting_value}_{annual_growth_rate}pct_{years}years.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("Please enter valid values for all inputs.")
    
    # Educational content
    st.subheader("📚 About Compound Growth")
    st.markdown("""
    **Compound growth** is the process where your investment gains earn additional gains over time. 
    This is often called "interest on interest" and is one of the most powerful concepts in investing.
    
    **Key Factors:**
    - **Starting amount**: The larger your initial investment, the more you'll have in the end
    - **Growth rate**: Even small differences in annual returns can have huge impacts over time
    - **Time horizon**: The longer you invest, the more compound growth works in your favor
    
    **Example**: \$10,000 at 7\% annual growth becomes \$38,697 in 20 years, but \$76,123 in 30 years!
    """)
