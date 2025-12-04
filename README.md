# 🐝 [W&B](https://docs.wandb.ai/) Weave YFinance Agent
<img src="figures/wb-cw.svg" alt="Weights and Biases by Coreweave" width="30%">

A beginner-friendly, public demo of a **financial research agent** built with:

- **Yahoo Finance (yfinance)** for market & fundamentals data.
- **MCP (Model Context Protocol)** tools (defined in `yfinance_server/tools_yfinance.py`).
- **OpenAI Agents SDK** for tool-using agent behaviour.
- **W&B Weave** for rich observability, tracing, and monitoring (LLMOps).
- **Streamlit** for a simple web UI.

Ask natural-language questions about stocks (prices, financial statements, holders, ESG, company profile) and the agent will call the right tools, reason, and answer.

---

## 📦 What’s in this repo?

- `yfinance_server/tools_yfinance.py`: MCP server that exposes **tools** for prices, charts, financial statements, holders, ESG, and profile info.  
- `yfinance_server/server.py`: Minimal programmatic runner: builds the agent, runs queries, and uses **Weave Scorers** and **OpenAI Agents SDK** output guardrails.
- `01_Quickstart_Agent.ipynb`: A notebook to test the agent.
- `streamlit_app/app.py`: A **Streamlit** app to chat with the agent in your browser.  
- `streamlit_app/utils_streamlit.py`: Helpers for rendering the agent’s markdown (including image support) and an “Open Weave Trace” button.

---

## 🧠 How it works (high level)

1. The *MCP server* (`yfinance_server/tools_yfinance.py`) defines Python tools using **yfinance**.  
2. The **OpenAI Agents SDK** agent (`yfinance_server/server.py`) connects to that server via **stdio** and can call those tools as needed.  
3. **Weave** (`yfinance_server/server.py`) logs each step (tool calls, outputs) so you can inspect runs interactively in your W&B project.  
4. The **Streamlit** UI (`streamlit_app/app.py`) is a thin layer that lets you choose a model and submit questions.

---

✅ Scoring and Output Guardrails 

This agent uses two output guardrails implemented with `W&B Weave Scorers` and `OpenAI Agents SDK Guardrails`:

1. **Hallucination Guardrail**: Uses a `weave.scorers.HallucinationFreeScorer` to detect unsupported or invented claims in the agent’s output, comparing it against the retrieved financial context.

2. **Numeric Consistency Guardrail**: Ensures all numeric values mentioned in the response match those present in the tool outputs (i.e. the retrieved financial context), using approximate numeric matching and a custom `weave.Scorer`, called `NumericConsistencyScorer`.

If either guardrail is triggered, the agent refuses to answer and logs the issue for monitoring via `Weave`.

---

## 🛠️ Tools exposed by `yfinance_server/tools_yfinance.py`

Below is the list parsed from the file, with a one-line summary:

- `get_historical_prices`: Fetch historical daily OHLCV prices for the given stock symbol over the specified period.
- `plot_price_history`: Generate a line plot of closing prices over time for the given symbol and period.
- `get_balance_sheet`: Get the latest balance sheet information for the given stock symbol and period.
- `get_income_statement`: Get the income statement data (annual) for the given stock symbol and period.
- `get_cash_flow`: Get the cash flow statement data (annual) for the given stock symbol and period.
- `get_financials`: Get the financials summary (annual) for the given stock symbol and period.
- `get_major_holders`: Get major holders information for the given stock symbol.
- `get_institutional_holders`: Get the top institutional holders for the given stock symbol.
- `get_sustainability`: Get the sustainability (ESG) scores and related metrics for the given stock symbol.
- `get_info`: Get key company information for the given stock symbol in an LLM-friendly structure.

> Each tool returns LLM-friendly data (lists/dicts) so the agent can summarize or format results for you.

### 💡 Example questions (one per tool)

- **get_historical_prices** → *What were the highest and lowest prices of AAPL in the last 6 months?*
- **plot_price_history** → *Plot the closing price history of AAPL over the past year.*
- **get_balance_sheet** → *Show Apple’s total assets and total liabilities for the latest fiscal year.*
- **get_income_statement** → *How much total revenue and net income did Microsoft report last year?*
- **get_cash_flow** → *What was Tesla’s cash generated from operating activities in the most recent year?*
- **get_financials** → *Summarize Amazon’s key financial figures (revenue, operating income, net income) for the last year.*
- **get_major_holders** → *What percentage of Apple’s shares are held by insiders and institutions?*
- **get_institutional_holders** → *Who are the top institutional holders of NVIDIA and how many shares do they own?*
- **get_sustainability** → *What are Microsoft’s ESG scores and are there any sustainability flags?*
- **get_info** → *Give me a concise company overview for Netflix (sector, market cap, employees, valuation ratios).*

You can paste these into the Streamlit app or run them via the notebook/script.

---

## 🚀 Quickstart

### 1) Environment

- Python 3.10+ recommended  
- Set your API keys (at least `OPENAI_API_KEY`).
- For `Weave` tracing, also set `WANDB_API_KEY`.

Create a `.env` file in the project root (or export env vars some other way):

```ini
OPENAI_API_KEY=sk-...            # required for OpenAI models
WANDB_API_KEY=...                # required for Weave traces
```

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

If you use virtual environments:
```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 3) Run the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

Then:
- Pick a model in the sidebar (from the presets listed in the app).  
- Ask a question like: **“Plot the closing price history of AAPL over the past year.”**  
- Watch the agent’s steps and open the **Weave trace** from the app.

### 4) Run from the terminal (no UI)

```bash
python server.py
```

Edit the `user_questions` list at the bottom of `server.py` to try your own queries.

### 5) Use programmatically (Jupyter/Script)

See **`01_Quickstart_Agent.ipynb`** in this repo for a minimal end‑to‑end example that uses `answer_questions` from `yfinance_server/server.py` to run queries programmatically.

---

## 📘 Notebook preview

The notebook shows how to:
- Call the `answer_questions` function (from `yfinance_server/server.py`) to automatically launch the MCP tools server and initialize the agent (using the default or specified model).
- Ask a question and read the final Markdown answer.
- (Optional) Display any generated plot image.

---

## 🔍 Troubleshooting

- **No data for ticker**: Check the symbol (e.g., `GOOG`, `AAPL`, `MSFT`); very new or illiquid tickers may have limited data.  
- **Model errors**: Make sure you’re using a model name that exists in your OpenAI account and that your API quota isn’t exceeded.  
- **Weave not logging**: Ensure you set `WANDB_API_KEY` and called `weave.init(project_name="<your_entity>/<your_project>")`. Note that Weave might log runs publicly by default for free users. If you want to change the `wandb` entity and project from the default values, you can add two environment variables to your `.env` file: `WANDB_ENTITY` and `WANDB_PROJECT`.
- **Agent stuck or slow**: The agent might be waiting on the tools. Check `yfinance_server/tools_yfinance.py` for any print statements (they can block MCP communication). Ensure `yfinance` calls aren’t hitting rate limits.  
- **Guardrail triggered**: The agent refused to answer due to guardrails (e.g., a predictive question). Adjust the question or remove that guardrail if needed (not recommended for real use).

---

## 🧾 License

MIT.

---

## 🙌 Acknowledgments

- [Weights and Biases Weave](https://docs.wandb.ai/weave).
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).
- [YFinance](https://pypi.org/project/yfinance/).
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol).