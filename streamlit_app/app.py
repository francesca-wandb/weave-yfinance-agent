import asyncio
import logging
import os
import sys
from contextlib import suppress
from io import StringIO
from pathlib import Path
from typing import Awaitable, Optional, TypeVar

import streamlit as st
import weave
from agents import (
    Agent,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from agents.mcp import MCPServerStdio
from agents.result import RunResult
from dotenv import load_dotenv
from weave.trace.call import Call

# Add project root to sys.path so all modules are importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from streamlit_app.utils_streamlit import (
    display_log_and_weave_button,
    render_markdown_with_images,
)
from yfinance_server.server import (
    GUARDRAIL_MESSAGE,
    MyRunContext,
    hallucination_guardrail,
    instructions_financial_assistant,
    numeric_consistency_guardrail,
)

# -----------------------------
# Early setup
# -----------------------------
load_dotenv()  # Load env for keys

# Avoid duplicate handler attachment on Streamlit reruns
_LOG_HANDLER_KEY = "_streamlit_log_handler_attached"
if "_log_buffer" not in st.session_state:
    st.session_state._log_buffer = StringIO()


class _StreamlitLogHandler(logging.StreamHandler):
    def __init__(self, buffer: StringIO) -> None:
        super().__init__(buffer)
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))


if not st.session_state.get(_LOG_HANDLER_KEY):
    handler = _StreamlitLogHandler(st.session_state._log_buffer)
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    st.session_state[_LOG_HANDLER_KEY] = True

# Initialize  Weights & Biases Weave once per session (and keep the client around)
if "_weave_client" not in st.session_state:
    # Ensure all prompts are published to Weave
    try:
        # Get default entity and project
        default_entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
        default_project = os.environ.get("WANDB_PROJECT", "weave-yfinance-agent")

        # Initialize W&B Weave client early with default entity/project
        st.session_state._weave_client = weave.init(f"{default_entity}/{default_project}")
    except Exception as e:
        logging.warning(f"Error initializing Weave client: {str(e)}")

st.set_page_config(
    page_title="🐝 W&B 🐝 Weave YFinance Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Constants & paths
# -----------------------------
MODEL_OPTIONS = [
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5",
    "gpt-4.1",
]
DEFAULT_MODEL = "gpt-5-mini"

CURRENT_DIR = Path(os.getcwd())
SERVER_SCRIPT = CURRENT_DIR / "yfinance_server" / "tools_yfinance.py"

if not SERVER_SCRIPT.exists():
    st.sidebar.warning(
        "YFinance MCP tools script not found at 'yfinance_server/tools_yfinance.py'.\n"
        "This app assumes it's available elsewhere as noted."
    )

# -----------------------------
# Async helpers
# -----------------------------

T = TypeVar("T")


def _get_loop() -> asyncio.AbstractEventLoop:
    """Create (once) and reuse a private event loop stored in session_state.
    Using a single loop avoids creating a new loop every rerun.
    """
    if "_loop" not in st.session_state:
        st.session_state._loop = asyncio.new_event_loop()
    return st.session_state._loop


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine on the app's private loop."""
    loop = _get_loop()
    return loop.run_until_complete(coro)


# ------------------------------
# MCP server lifecycle (cached)
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_mcp_server() -> MCPServerStdio:
    """Start the MCP server once per Streamlit session and keep it running."""

    async def _start():
        mcp = await MCPServerStdio(
            name="YFinance MCP Server",
            params={"command": "python", "args": [str(SERVER_SCRIPT)]},
        ).__aenter__()
        return mcp

    return run_async(_start())


# ----------------------------------
# Agent builder (cheap, not cached)
# ----------------------------------
def build_agent(model_name: str, mcp_server: MCPServerStdio) -> Agent:
    """Construct a FinancialAssistant Agent with the given model and MCP server."""
    return Agent(
        name="FinancialAssistant",
        instructions=instructions_financial_assistant,
        mcp_servers=[mcp_server],
        model=model_name,
        output_guardrails=[hallucination_guardrail, numeric_consistency_guardrail],
    )


# -----------------------------
# Sidebar: model selection
# -----------------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL

st.sidebar.header("Model")
st.sidebar.caption("Choose which model the Agent should use.")
selected_model = st.sidebar.selectbox(
    "Model",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(st.session_state.selected_model),
)
apply_model_clicked = st.sidebar.button("Apply Model", use_container_width=True)

# Ensure MCP server exists
mcp_server = get_mcp_server()

# Build or rebuild the agent on demand
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(st.session_state.selected_model, mcp_server)

if apply_model_clicked:
    st.session_state.selected_model = selected_model
    with st.sidebar:
        try:
            st.session_state.agent = build_agent(selected_model, mcp_server)
            st.success(f"Model set to {selected_model}")
        except Exception as e:
            st.error(f"Failed to update model: {e}")


# -----------------------------
# Header
# -----------------------------
col_title, col_img = st.columns([4, 1])
with col_title:
    st.title("🐝 W&B 🐝 Weave YFinance Agent")
    st.markdown("Based on [YFinance](https://pypi.org/project/yfinance/)")
with col_img:
    logo_path = CURRENT_DIR / "wandb_logo.png"
    if logo_path.exists():
        st.image(str(logo_path))

st.info(f"**Current model:** `{st.session_state.selected_model}`")

# -----------------------------
# Examples
# -----------------------------
with st.expander("💡 Example questions", expanded=False):
    example_questions_md = (
        '- Historical Price Data – "What were the highest and lowest prices of AAPL in the last 6 months?"\n'
        '- Price Trend Chart – "Can you plot the stock price history of AAPL over the past year?"\n'
        '- Balance Sheet Inquiry – "What are Apple\'s total assets and liabilities in its latest balance sheet?"\n'
        '- Income Statement Inquiry – "How much revenue and net income did Apple report last year?"\n'
        '- Cash Flow Inquiry – "How much cash did Apple generate from operating activities in the most recent year?"\n'
        "- Financials Summary – \"Give me a summary of Apple's key financial figures for the last year "
        '(revenue, profit, etc.)"\n'
        '- Major Holders – "Who are the major shareholders of Apple and what percentage do they hold?"\n'
        '- Institutional Holders – "Which institutions hold the most Apple stock, and how many shares do they own?"\n'
        '- ESG/Sustainability – "What are Apple\'s environmental, social, and governance (ESG) scores and ratings?"\n'
        '- Company Profile – "Tell me about Apple Inc. (sector, market cap, employees, etc.)"'
    )
    st.markdown(example_questions_md, unsafe_allow_html=True)

# -----------------------------
# Main input
# -----------------------------
user_query = st.text_input(
    "Ask a financial question:",
    help="Questions are sent to the Agent, which can call YFinance tools via MCP.",
)
run_clicked = st.button("Run Query")

# -----------------------------
# Run agent
# -----------------------------
if run_clicked and user_query.strip():

    @weave.op(name="Agent Workflow")
    async def _run(agent: Agent, user_query: str, ctx: MyRunContext) -> tuple[Optional[RunResult], Call]:
        """Run the agent for a given user question and context."""

        call = weave.require_current_call()
        result = None
        try:
            result = await Runner.run(starting_agent=agent, input=user_query, context=ctx)
            # Add Weave reaction to the call
            st.session_state._feedback_ok = True
            call.feedback.add_reaction("✅")
        except OutputGuardrailTripwireTriggered as guardrail_error:
            st.error(f"⚠️ {GUARDRAIL_MESSAGE}")
            # Add Weave reaction to the call
            st.session_state._feedback_ok = False
            call.feedback.add_reaction("❌")

            details: dict = {}
            for key in ("guardrail_name", "output_info", "message", "details", "reason"):
                val = getattr(guardrail_error, key, None)
                if val is not None:
                    details[key] = val
            if not details and getattr(guardrail_error, "args", None):
                details["args"] = [str(a) for a in guardrail_error.args]
            with suppress(Exception):
                st.json(details)

        except Exception as e:
            st.error(f"Agent error: {e}")

        return result, call

    with st.spinner("Agent is processing..."):
        # Create run context
        ctx = MyRunContext(user_question=user_query)
        # Main call
        result, call = run_async(_run(st.session_state.agent, user_query, ctx))
        st.session_state._last_result = result
        st.session_state._feedback_ok = True

        # Display Weave button and log call guardrail feedbaack to Weave
        try:
            # Add Weave URL from logging
            display_log_and_weave_button(call.ui_url)

        except Exception:
            pass

    # ---------------------------
    # Build progress logs safely
    # ---------------------------
    progress_logs = []
    final_md = None
    if result is not None:
        with suppress(Exception):
            for item in getattr(result, "new_items", []) or []:
                agent_name = getattr(getattr(item, "agent", None), "name", "Agent")
                if getattr(item, "type", "") == "tool_call_item":
                    progress_logs.append(f"**{agent_name}**: Calling the tool: `{item.raw_item.name}`")
                elif getattr(item, "type", "") == "tool_call_output_item":
                    output_str = str(getattr(item, "output", ""))
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "..."
                    progress_logs.append(f"**{agent_name}**: Tool output: `{output_str}`")
                elif getattr(item, "type", "") == "message_output_item":
                    from agents import ItemHelpers

                    text = ItemHelpers.text_message_output(item)
                    progress_logs.append(f"**{agent_name}**: {text}")
        final_md = getattr(result, "final_output", None)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # Two columns: Answer | Flow
    # -------------------------
    col1, space_col, col2 = st.columns([1, 0.1, 1])

    with col1:
        st.subheader("Agent's Answer")
        if final_md:
            render_markdown_with_images(final_md, image_base_path=".")
        else:
            st.info("No final answer produced.")

    with col2:
        st.subheader("Agent's Flow")
        if progress_logs:
            for log in progress_logs[:-1] if len(progress_logs) > 1 else progress_logs:
                st.markdown(f"- {log}")
        else:
            st.info("No intermediate steps recorded.")
