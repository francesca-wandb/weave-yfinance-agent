import base64
import io
import logging
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("YFinance Server")

# Setup logging for the server
log_file = os.path.join(os.getcwd(), "results", f"yfinance_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("YFinance MCP server started.")


def select_period_columns(df: pd.DataFrame, period: str) -> list[str]:
    """
    Select the appropriate column(s) from a financial DataFrame based on the specified time period.

    The financial statement DataFrames in yfinance (e.g. balance_sheet, income_stmt, cashflow, financials)
    use columns as dates/periods (annual by default) and rows as accounts/line items.
    This helper converts the column labels to timestamps and supports:
      - 'latest' or 'nth latest' selection.
      - A 4-digit year 'YYYY'.
      - An exact date (e.g. 'YYYY-MM-DD'),
      - 'all' to return all period columns.

    :param df: The financial DataFrame containing date-indexed columns.
    :param period: The time period to select (e.g. 'latest', 'nth latest', 'YYYY', 'YYYY-MM-DD', or 'all').
    :return: A list of column names corresponding to the selected period.
    """
    # Parse column labels to timestamps (for robust date comparisons)
    try:
        col_dates = pd.to_datetime(df.columns)  # Convert column labels to datetime
    except Exception as e:
        raise ValueError(f"Failed to parse DataFrame columns as dates: {e}")

    # Keep original labels alongside parsed dates to preserve exact column keys when returning
    cols_list = list(df.columns)

    # If no columns or all parse to NaT, the DataFrame is not in the expected financial statements format
    if col_dates.isnull().all():
        raise ValueError("No valid date columns found in DataFrame.")

    # Normalize case for string
    period = period.lower() if isinstance(period, str) else period

    # 1) Return all columns when explicitly requested (used for bulk inspection/exports)
    if period == "all":
        return [str(col) for col in cols_list]

    # 2) Latest or nth-latest selection by sorting dates ascending and picking from the end
    # e.g. 'latest', '2nd latest', '10th latest'
    nth_match = re.match(r"^(\d+)(st|nd|rd|th)\s+latest$", period)
    if period == "latest" or nth_match:
        n = 1
        if nth_match:
            n = int(nth_match.group(1))
        # Sort dates in ascending order and pick the n-th from last (latest = n=1)
        sorted_idx = sorted(range(len(col_dates)), key=lambda i: col_dates[i])  # indices sorted by date
        if n > len(sorted_idx):
            raise ValueError(f"Requested {n}th latest period, but only {len(sorted_idx)} periods available.")
        # Pick the n-th latest (i.e., -n index in sorted list)
        col_index = sorted_idx[-n]
        return [str(cols_list[col_index])]

    # 3) 4-digit Year selection (matches any column whose parsed year equals the requested year)
    if isinstance(period, str) and re.fullmatch(r"\d{4}", period):
        year = int(period)
        matches = [str(cols_list[i]) for i, ts in enumerate(col_dates) if ts.year == year]
        if not matches:
            # Provide available years in error message for clarity
            available_years = sorted({ts.year for ts in col_dates if not pd.isna(ts)})
            raise ValueError(f"No columns match year {year}. Available years: {available_years}")
        return matches

    # 4) Exact date match (rare but supported for completeness)
    try:
        target_date = pd.to_datetime(period)
    except Exception:
        target_date = None
    if pd.Timestamp(target_date) in col_dates.values:
        # Find all exact matches (though typically there would be at most one)
        matches = [str(cols_list[i]) for i, ts in enumerate(col_dates) if ts == target_date]
        return matches

    # If none of the above matched, raise a clear error with usage guidance
    raise ValueError(
        f"Period '{period}' is not recognized or not available. "
        f"Use 'latest', 'nth latest', a 4-digit year, a full date (YYYY-MM-DD), or 'all'."
    )


# 1. Tool: Get historical daily prices
@mcp.tool()
def get_historical_prices(symbol: str, period: str = "1y") -> dict:
    """
    Fetch historical daily OHLCV prices for the given stock symbol over the specified period.

    The DataFrame retrieved contains daily price data with columns for Open, High, Low, Close, and Volume (OHLCV),
    capturing each day's trading information for the specified period.

    Notes:
    - Uses yf.Ticker(symbol).history(period=...) to pull daily prices.
    - Returns a JSON-serializable structure (list of dicts).

    :param symbol: Stock ticker symbol (e.g. "GOOG").
    :param period: Lookback period (e.g. "1y" for 1 year, "6mo" for 6 months). Defaults to 1 year.
    :return: Dictionary with symbol and a list of daily price data (date, open, high, low, close, volume).
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the history DataFrame
        hist = ticker.history(period=period)
        if hist.empty:
            return {"error": f"No historical data found for {symbol} over period {period}."}
        # Reset index to get dates as a column and format as ISO string for portability
        hist = hist.reset_index()
        data_points = []
        for _, row in hist.iterrows():
            data_points.append(
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )
        return {"symbol": symbol, "period": period, "historical": data_points}
    except Exception as e:
        logger.error("Error in get_historical_prices for %s: %s", symbol, e)
        return {"error": str(e)}


# 2. Tool: Plot historical price trend
@mcp.tool()
def plot_price_history(symbol: str, period: str = "1y") -> str:
    """
    Generate a line plot of closing prices over time for the given symbol and period.

    This function uses daily historical pricing data
    (OHLCV format: Open, High, Low, Close, Volume) for the specified period,
    focusing on the 'Close' price column to plot the stock's closing price trend over time.

    Implementation details:
    - Uses matplotlib to plot Close prices.
    - Saves the plot into a PNG file and returns a confirmation string with the filename.
    - The image is also base64-encoded internally to keep the pipeline flexible.

    :param symbol: Stock ticker symbol.
    :param period: Lookback period accepted by yfinance.history (e.g. '1y', '6mo').
    :return: Path/filename info (string) or error message.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the history DataFrame
        hist = ticker.history(period=period)
        if hist.empty:
            logger.warning("No data for %s in plot_price_history", symbol)
            return f"Unable to retrieve data for {symbol}."
        # Plot the closing price
        plt.figure(figsize=(6, 4))
        hist["Close"].plot(title=f"{symbol} price history ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        # Save plot to a PNG in memory and base64 encode (for potential JSON transport)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        # We could return the image encoded, but to keep it simple, save to file instead
        filename = f"{symbol}_price_{period}.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(img_base64))
        return f"Saved price plot for {symbol} as {filename}."
    except Exception as e:
        logger.error("Error in plot_price_history for %s: %s", symbol, e)
        return f"Failed to plot data for {symbol}: {e}"


# 3. Tool: Get latest balance sheet info
@mcp.tool()
def get_balance_sheet(symbol: str, period: str = "latest") -> list[dict]:
    """
    Get the latest balance sheet information for the given stock symbol and period.

    The balance sheet data is annual by default,
    with columns representing year-end dates for each fiscal year
    and rows as balance sheet accounts.
    Key accounts include the following:
    - Current and non-current assets (e.g. cash, accounts receivable, inventory, total assets).
    - Liabilities (e.g. accounts payable, short-term and long-term debt, total liabilities).
    - Shareholders’ equity (e.g. common stock, retained earnings, total stockholder equity).

    Data source:
    - yf.Ticker(symbol).balance_sheet (annual by default: typically last 4 years).
      Rows = accounts (e.g. Total Assets), Columns = reporting dates (annual).

    Output format:
    - List of dicts: {"account": <row_name>, "<period_col>": <value>, ...}

    Filtering:
    - Drops rows that are entirely 0/NaN in the selected period(s) for conciseness.

    :param symbol: Stock ticker symbol (e.g. "GOOG").
    :param period: 'latest', 'nth latest', 'YYYY', 'YYYY-MM-DD', or 'all'. Defaults to 'latest'.
    :return: A list of dictionaries for selected period(s).
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the balance sheet DataFrame
        df = ticker.balance_sheet
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("No data for %s in get_balance_sheet", symbol)
            return []
        # Select the relevant period column(s) using the helper
        cols = select_period_columns(df, period)
        data_subset = df[cols]
        # Filter out accounts (rows) that have all zero or NaN values in the selected columns
        mask = (data_subset.fillna(0) != 0).any(axis=1)
        filtered = data_subset[mask]
        # Format as list of account records
        records = []
        for account, row in filtered.iterrows():
            record = {"account": account}
            for col in cols:
                # Use original values; replace NaN with 0 for clarity in output
                value = row[col]
                if pd.isna(value):
                    value = 0
                # Ensure JSON-serializable native Python types
                if isinstance(value, numpy.generic):
                    value = value.item()
                record[col] = float(value) if isinstance(value, (int, float)) else value
            records.append(record)
        return records
    except Exception as e:
        logger.error("Error in get_balance_sheet for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 4. Tool: Get income statement info (annual)
@mcp.tool()
def get_income_statement(symbol: str, period: str = "latest") -> list[dict]:
    """
    Get the income statement data (annual) for the given stock symbol and period.

    The income statement DataFrame is annual,
    with each column representing a year (fiscal year)
    and each row representing an income statement line item.
    Key fields include the following:
    - Major revenue and expense metrics such as total revenue.
    - Cost of revenue.
    - Gross profit.
    - Operating expenses.
    - Operating income (or EBIT).
    - Pre-tax income.
    - Net income.
    They give a yearly overview of the company's profitability.

    Data source:
    - yf.Ticker(symbol).income_stmt (annual).
        Similar shape to .financials but matches Yahoo's "Income Statement" canonical table.

    Notes:
    - Period selection handled via `select_period_columns` (see helper above).
    - Filters out lines that are all 0/NaN for the chosen period(s).
    - Returns LLM-friendly list of dictionaries.

    :param symbol: Stock ticker symbol.
    :param period: 'latest', 'nth latest', 'YYYY', 'YYYY-MM-DD', or 'all'. Defaults to 'latest'.
    :return: List of dicts where each dict is an income statement line item with values per selected period.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the income statement DataFrame
        df = getattr(ticker, "income_stmt", None)
        if df is None:
            logger.warning("yf.Ticker has no income_stmt attribute, using financials instead")
            df = ticker.financials
            if df is None or (hasattr(df, "empty") and df.empty):
                logger.info("No data for %s in get_income_statement", symbol)
                return []
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("No data for %s in get_income_statement", symbol)
            return []
        cols = select_period_columns(df, period)
        data_subset = df[cols]
        mask = (data_subset.fillna(0) != 0).any(axis=1)
        filtered = data_subset[mask]
        records = []
        for item, row in filtered.iterrows():
            record = {"item": item}
            for col in cols:
                value = row[col]
                if pd.isna(value):
                    value = 0
                if isinstance(value, numpy.generic):
                    value = value.item()
                record[col] = float(value) if isinstance(value, (int, float)) else value
            records.append(record)
        return records
    except Exception as e:
        logger.error("Error in get_income_statement for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 5. Tool: Get cash flow statement info (annual)
@mcp.tool()
def get_cash_flow(symbol: str, period: str = "latest") -> list[dict]:
    """
    Get the cash flow statement data (annual) for the given stock symbol and period.

    This annual cash flow DataFrame includes columns for each year and rows for each cash flow line item.
    Key fields encompass net cash from operating activities:
    - Net income and including adjustments like depreciation and changes in working capital.
    - Net cash from investing activities (e.g. capital expenditures and acquisitions or sales of assets).
    - Net cash from financing activities (e.g. issuance or repayment of debt, stock issuance/buybacks, dividends).
    - The net change in cash for the period.

    Data source:
    - yf.Ticker(symbol).cashflow (annual). Includes operating, investing, financing cash flows, etc.

    Implementation:
    - Uses the same selection/filtering pipeline as balance sheet & income statement.
    - Converts values to native Python types for JSON compatibility.

    :param symbol: Stock ticker symbol.
    :param period: 'latest', 'nth latest', 'YYYY', 'YYYY-MM-DD', or 'all'. Defaults to 'latest'.
    :return: List of dicts representing cash flow line items for selected period(s).
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the cash flow DataFrame
        df = ticker.cashflow
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("No data for %s in get_cash_flow", symbol)
            return []
        cols = select_period_columns(df, period)
        data_subset = df[cols]
        mask = (data_subset.fillna(0) != 0).any(axis=1)
        filtered = data_subset[mask]
        records = []
        for item, row in filtered.iterrows():
            record = {"item": item}
            for col in cols:
                value = row[col]
                if pd.isna(value):
                    value = 0
                if isinstance(value, numpy.generic):
                    value = value.item()
                record[col] = float(value) if isinstance(value, (int, float)) else value
            records.append(record)
        return records
    except Exception as e:
        logger.error("Error in get_cash_flow for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 6. Tool: Get financials summary (Yahoo Finance "Financials" table, annual)
@mcp.tool()
def get_financials(symbol: str, period: str = "latest") -> list[dict]:
    """
    Get the financials summary (annual) for the given stock symbol and period.

    The financials DataFrame returned is an annual summary resembling an income statement.
    Columns correspond to each fiscal year (annual reporting periods),
    and rows include major financial results such as the following:
    - Total revenue.
    - Cost of revenue.
    - Gross profit.
    - Operating income.
    - Net income.

    Data source:
    - yf.Ticker(symbol).financials (annual).
        This is essentially an income-statement-like table for the last few years.

    Output:
    - List of dicts with {"item": <line_name>, "<period_col>": <value>, ...}
      for easy agent consumption and downstream processing.

    :param symbol: Stock ticker symbol.
    :param period: 'latest', 'nth latest', 'YYYY', 'YYYY-MM-DD', or 'all'. Defaults to 'latest'.
    :return: List of dicts for selected period(s).
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the annual financials (income statement-like) DataFrame
        df = ticker.financials
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("No data for %s in get_financials", symbol)
            return []
        cols = select_period_columns(df, period)
        data_subset = df[cols]
        mask = (data_subset.fillna(0) != 0).any(axis=1)
        filtered = data_subset[mask]
        records = []
        for item, row in filtered.iterrows():
            record = {"item": item}
            for col in cols:
                value = row[col]
                if pd.isna(value):
                    value = 0
                if isinstance(value, numpy.generic):
                    value = value.item()
                record[col] = float(value) if isinstance(value, (int, float)) else value
            records.append(record)
        return records
    except Exception as e:
        logger.error("Error in get_financials for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 7. Tool: Get major holders data
@mcp.tool()
def get_major_holders(symbol: str) -> list[dict]:
    """
    Get major holders information for the given stock symbol.

    The major holders data is a small table of key ownership statistics.
    It typically consists of two columns (value and description) for each entry.
    Important fields provided include the following (among other major shareholding metrics):
    - The percentage of shares held by insiders.
    - The percentage of shares held by institutions.
    - The number of institutions holding shares.

    Data source:
    - yf.Ticker(symbol).major_holders (small DataFrame). Typically includes values like:
      '% of Shares Held by Insiders', '% of Shares Held by Institutions',
      'Number of Institutions Holding Shares', etc.

    Normalization:
    - Attempts to parse percentages to numeric (without the % sign) and K/M/B shorthand to numbers.
    - Converts numpy/pandas scalar types to native Python types for JSON serialization.

    :param symbol: Stock ticker symbol.
    :return: A list of dictionaries: [{"metric": <description>, "value": <parsed_value>}, ...]
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the major holders DataFrame
        df = ticker.major_holders
        if df is None or df.empty:
            logger.info("No major holders data for %s", symbol)
            return []
        records = []
        # The DataFrame is expected to have two columns: [value, description]
        for _, row in df.iterrows():
            if len(row) >= 2:
                raw_value = row[0]
                description = str(row[1])
            else:
                # If only one column is present for some reason, use index as description
                raw_value = row.iloc[0]
                description = str(row.name)
            # Clean and convert the value (percentages, K/M/B shorthand, comma separators)
            value = raw_value
            if isinstance(value, str):
                val_str = value.strip()
                # Remove percentage sign and parse as float
                if val_str.endswith("%"):
                    try:
                        value = float(val_str.replace("%", ""))
                    except Exception:
                        value = val_str
                # Handle commas and scale suffixes
                elif any(ch in val_str for ch in [",", "K", "M", "B"]):
                    factor = 1.0
                    if val_str.endswith("K"):
                        factor = 1e3
                        val_str = val_str[:-1]
                    elif val_str.endswith("M"):
                        factor = 1e6
                        val_str = val_str[:-1]
                    elif val_str.endswith("B"):
                        factor = 1e9
                        val_str = val_str[:-1]
                    try:
                        val_num = float(val_str.replace(",", "")) * factor
                        value = int(val_num) if val_num.is_integer() else val_num
                    except Exception:
                        value = val_str
            # Convert numpy types to Python types
            if isinstance(value, numpy.generic):
                value = value.item()
            records.append({"metric": description, "value": value})
        return records
    except Exception as e:
        logger.error("Error in get_major_holders for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 8. Tool: Get institutional holders data
@mcp.tool()
def get_institutional_holders(symbol: str) -> list[dict]:
    """
    Get the top institutional holders for the given stock symbol.

    The institutional holders data is a table listing major institutional shareholders and their holdings.
    It contains columns such as
    - "Holder" (the institution name).
    - "Shares" (number of shares held).
    - "Date Reported" (the filing date of the holding).
    - "% Out" (the percentage of outstanding shares that this holding represents).
    - "Value" (the market value of the shares held).
    Each row of the DataFrame corresponds to one top institutional holder and their stake in the company.

    Data source:
    - yf.Ticker(symbol).institutional_holders (DataFrame of holders with columns like
      'Holder', 'Shares', 'Date Reported', '% Out', 'Value', etc.).

    Normalization:
    - Parses percentages to floats (without the % sign).
    - Converts comma-separated numbers and K/M/B shorthand into numeric values.
    - Formats pandas Timestamps to 'YYYY-MM-DD'.

    :param symbol: Stock ticker symbol.
    :return: A list of dictionaries, each representing one institutional holder's row.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the institutional holders DataFrame
        df = ticker.institutional_holders
        if df is None or df.empty:
            logger.info("No institutional holders data for %s", symbol)
            return []
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                # Skip NaN values to keep payload compact
                if pd.isna(val):
                    continue
                if isinstance(val, str):
                    val_str = val.strip()
                    # Parse percentage strings -> float
                    if val_str.endswith("%"):
                        try:
                            val = float(val_str.replace("%", ""))
                        except Exception:
                            val = val_str
                    # Remove commas and parse K/M/B shorthands
                    elif "," in val_str or val_str.endswith(("K", "M", "B")):
                        factor = 1.0
                        if val_str.endswith("K"):
                            factor = 1e3
                            val_str = val_str[:-1]
                        elif val_str.endswith("M"):
                            factor = 1e6
                            val_str = val_str[:-1]
                        elif val_str.endswith("B"):
                            factor = 1e9
                            val_str = val_str[:-1]
                        try:
                            val_num = float(val_str.replace(",", "")) * factor
                            val = int(val_num) if val_num.is_integer() else val_num
                        except Exception:
                            val = val_str
                # Convert pandas Timestamp to ISO date for readability/stability
                if isinstance(val, pd.Timestamp):
                    val = val.strftime("%Y-%m-%d")
                # Convert numpy scalar types to native Python
                if isinstance(val, numpy.generic):
                    val = val.item()
                record[col] = val
            # Only add non-empty records
            if record:
                records.append(record)
        return records
    except Exception as e:
        logger.error("Error in get_institutional_holders for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 9. Tool: Get sustainability (ESG) data
@mcp.tool()
def get_sustainability(symbol: str) -> list[dict]:
    """
    Get the sustainability (ESG) scores and related metrics for the given stock symbol.

    The sustainability DataFrame (if available) provides a single row of various ESG metrics and flags.
    Its columns include the company’s environmental, social, and governance scores
    (e.g. environmentScore, socialScore, governanceScore) along with a total ESG score,
    the ESG performance rating (such as 'UNDER_PERF' or similar indicating overall ESG performance),
    peer group information and percentile rankings (how the company compares to peers on ESG measures),
    and boolean flags indicating involvement in specific controversial categories
    (for example, palmOil, gambling, nuclear, tobacco,
    and other issues are False/True depending on the company's activities).
    These fields give insight into the company's sustainability profile as reported by Yahoo Finance.

    Data source:
    - yf.Ticker(symbol).sustainability (single-row DataFrame with ESG metrics when available).

    Output:
    - List of {"metric": <name>, "value": <value>} entries, skipping missing values,
      with native Python types for JSON compatibility.

    :param symbol: Stock ticker symbol.
    :return: List of metric dicts or empty list if not available.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the sustainability DataFrame (usually a single-row DataFrame of ESG metrics)
        df = ticker.sustainability
        if df is None or df.empty:
            logger.info("No sustainability data for %s", symbol)
            return []
        records = []
        # If the DataFrame has one row, iterate over its items (ESG datasets are typically single-row)
        series = df.iloc[0]
        for key, val in series.items():
            if pd.isna(val):
                continue
            # Convert numpy/pandas specific scalars to native types
            if isinstance(val, numpy.generic):
                val = val.item()
            if isinstance(val, pd.Timestamp):
                val = val.strftime("%Y-%m-%d")
            records.append({"metric": str(key), "value": val})
        return records
    except Exception as e:
        logger.error("Error in get_sustainability for %s: %s", symbol, e)
        return [{"error": str(e)}]


# 10. Tool: Get key company info (dictionary -> LLM-friendly list of fields)
@mcp.tool()
def get_info(symbol: str) -> list[dict]:
    """
    Get key company information for the given stock symbol in an LLM-friendly structure.

    This function retrieves a selection of fundamental company information fields from Yahoo Finance.
    Key fields returned include the company's long name and short name, its sector and industry,
    the country of headquarters, the number of full-time employees, the reporting currency for financials,
    market capitalization, enterprise value, the current stock price, important valuation ratios
    (such as trailing PE, forward PE, PEG ratio, and price-to-book ratio),
    the stock’s beta (volatility relative to the market),
    the dividend yield (current), the five-year average dividend yield,
    and a long business summary describing the company’s operations.
    These fields provide a concise profile of the company’s identity, size, valuation, and business scope.

    Data source:
    - yf.Ticker(symbol).info (dict).
        As it may include hundreds of fields,
        we select a concise, high-signal subset useful for profiling and screening.

    Formatting:
    - Returns [{"field": <name>, "value": <value>}, ...] with native Python types.

    :param symbol: Stock ticker symbol.
    :return: List of selected info fields and values, suitable for downstream agent consumption.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch the dictionary of company information
        info = ticker.info
        if not info:
            logger.info("No company info available for %s", symbol)
            return []
        # Select key fields to include (profile, valuation, and summary fields)
        key_fields = [
            "longName",
            "shortName",
            "sector",
            "industry",
            "country",
            "fullTimeEmployees",
            "financialCurrency",
            "marketCap",
            "enterpriseValue",
            "currentPrice",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToBook",
            "beta",
            "dividendYield",
            "fiveYearAvgDividendYield",
            "longBusinessSummary",
        ]
        records = []
        for field in key_fields:
            if field in info:
                val = info[field]
                # Normalize numpy types (e.g., numpy.int64/float64) to native Python for JSON
                if isinstance(val, numpy.generic):
                    val = val.item()
                # Normalize numpy.bool_ to bool, leave bool as-is
                if isinstance(val, (numpy.bool_, bool)):
                    val = bool(val)
                # Normalize numpy numeric scalars to float/int
                if isinstance(val, (numpy.float64, numpy.float32)):
                    val = float(val)
                if isinstance(val, (numpy.int64, numpy.int32)):
                    val = int(val)
                records.append({"field": field, "value": val})
        return records
    except Exception as e:
        logger.error("Error in get_info for %s: %s", symbol, e)
        return [{"error": str(e)}]


# If run directly (for development), launch the server.
# In practice, we'll launch via the OpenAI agent (MCPServerStdio), so this is optional:
if __name__ == "__main__":
    # Start the MCP server loop
    mcp.run()
