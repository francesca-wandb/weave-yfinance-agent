"""
This agent orchestrates an OpenAI-based financial research assistant
with MCP tools and Weights and Biases Weave for observability, tracing, and monitoring.
It configures logging, initializes Weave, defines output guardrails,
and runs example queries via an Agent and an MCP stdio server.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import nest_asyncio
import weave
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    Runner,
    output_guardrail,
)
from agents.lifecycle import AgentHooksBase
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from weave.scorers import HallucinationFreeScorer
from weave.trace.call import Call

# Add project root to sys.path so all modules are importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yfinance_server.utils_server import _ensure_output_dir, _fraction_matched, _numbers, setup_logging

load_dotenv()

# Apply nest_asyncio at the module level to allow nested event loops
nest_asyncio.apply()

MODEL = "gpt-5-mini"
RESULTS_DIR = "results"


# Initialize output directory and logging
OUTPUT_DIR = _ensure_output_dir(os.getcwd(), RESULTS_DIR)

# Setup logging
logger, _ = setup_logging()

# Initialize  Weights & Biases Weave once per session (and keep the client around)
try:
    # Get default entity and project
    default_entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
    default_project = os.environ.get("WANDB_PROJECT", "weave-yfinance-agent")

    # Initialize W&B Weave client early with default entity/project
    weave_client = weave.init(f"{default_entity}/{default_project}")
except Exception as e:
    logger.warning(f"Error initializing Weave client: {str(e)}")

today = datetime.today()
formatted_date = today.strftime("%B %d, %Y")

instructions_financial_assistant = f"""
You are an AI **financial research assistant** operating within an OpenAI Agent environment
that supports structured tool and function calling via the MCP (Model Context Protocol) system.
You have access to Python-based financial tools, including the `yfinance` library,
and may invoke these tools via function calls to retrieve, analyze, and visualize market data.
The current date is {formatted_date}.

### Core Capabilities
- Use the available MCP tools or Python functions to:
  * Retrieve stock, ETF, or index data (price history, returns, fundamentals, etc.).
  * Compute statistics (mean, median, volatility, max drawdown, etc.).
  * Generate charts, summaries, or comparisons between instruments.
  * Accurately handle both daily and period-based financial data.

### Period Formatting Rules
When interpreting time periods, use the following formats:
- `'latest'`: The most recent available data point.
- `'nth latest'`: e.g., `'2nd latest'`, `'10th latest'`: the nth most recent period.
- `'YYYY'`: A 4-digit year, e.g., `'2025'`, representing all data from that calendar year.
- `'YYYY-MM-DD'`: An exact date in ISO format.
- `'all'`: Include all available data.
If a period does not match one of these formats, raise a clear error listing the valid options.

### Response Formatting and Style
- Always respond with structured, factual, data-backed output.
- Use Markdown formatting for clarity:
  * Tables for numerical data.
  * Bullet points for summaries.
- When presenting data:
  * Explicitly state what the data represents (e.g., "closing prices for the last 30 days").
  * Include relevant statistics (mean, min, max, standard deviation) when applicable.
  * Clearly label all units and timeframes.
  * Add charts if they enhance understanding.
  * Embed images using the Markdown format: `![Alt Text](image_filename)`.

### Behavioral Rules
- Use tools or function calls **whenever data retrieval or computation is needed**.
- If a user query is ambiguous, clearly state the key ambiguities and any assumptions you are making,
and do not request clarification.
- Do **not** speculate or provide unverified information.
- Minimize mathematical computations.
Perform calculations and report derived numerical results
only when the user query explicitly requires it,
otherwise report the raw data as provided by the tools as much as possible.
- You may use standard simplified numerical notation,
when it does not compromise the accuracy or interpretability of the results.
- Be consistent in the precision and notation of numerical results.
- Maintain a professional, analytical, and concise tone.
- Ensure every response is deterministic and explainable from the available data.
- Do not ask the user any follow-up questions.
- Do not exceed 500 words in your response.
"""


HALLUCINATION_PROMPT = """
Do not consider newly introduced ticker symbols as hallucinations,
nor minor deviations from brand or product names.
If numbers are slightly approximate and do not match exactly the ones in the context,
do not consider them as hallucinations.
Ignore currencies inferred from the context for your evaluation.
Ignore minor variations between the output and the context,
so only consider major hallucinations as hallucinations.
"""

GUARDRAIL_MESSAGE = "Response blocked by output guardrail."


@dataclass
class MyRunContext:
    """
    Context container used during a single agent run.

    Attributes:
        user_question: The original user question.
        tools: List of invoked tools.
        tool_outputs: Accumulated outputs returned by tools during the run.
        last_tool_output: The most recent tool output.
    """

    user_question: Optional[str] = None
    tools: Optional[List[Dict[str, str]]] = field(default_factory=list)
    tool_outputs: List[str] = field(default_factory=list)
    last_tool_output: Optional[str] = None


class CaptureToolOutputs(AgentHooksBase[MyRunContext, Any]):
    """Agent hook that captures tool outputs into the run context."""

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """
        Capture tool result when a tool finishes and store it in the context.

        Args:
            context: Wrapper around the current run context.
            agent: The running agent instance.
            tool: The tool instance that just finished.
            result: The tool result to capture.
        """
        ctx = context.context
        as_text = str(result)
        if tool.name is not None:
            ctx.tools.append({tool.name: tool.description})
        ctx.tool_outputs.append(as_text)
        ctx.last_tool_output = as_text


# Instantiate the Weave built-in Hallucination Scorer
hallucination_scorer = HallucinationFreeScorer(model_id=MODEL, temperature=1)
hallucination_scorer.system_prompt += HALLUCINATION_PROMPT


@output_guardrail
async def hallucination_guardrail(ctx: Any, agent: Any, agent_output: str) -> GuardrailFunctionOutput:
    """
    Guardrail that flags potential hallucinations using `Weave.scorers.HallucinationFreeScorer` scorer.

    Args:
        ctx: Run context wrapper providing access to `MyRunContext`.
        agent: The agent instance (unused).
        agent_output: The agent's textual output to validate.

    Returns:
        GuardrailFunctionOutput indicating whether hallucinations were detected.
    """
    context = ctx.context
    context_text = (
        f"User question: {context.user_question}\n"
        f"Data retrieved from Yahoo Finance (YFinance) corresponding to the user question: {context.tool_outputs} "
        f"using tools: {str(context.tools)}"
    )
    score: Dict[str, Any] = asyncio.run(hallucination_scorer.score(context=context_text, output=agent_output))

    has_hallucination = bool(score.get("has_hallucination", False))

    return GuardrailFunctionOutput(
        output_info={"hallucination_detected": has_hallucination},
        tripwire_triggered=has_hallucination,
    )


class NumericScore(TypedDict):
    """Typed dict for numeric consistency scorer output."""

    fraction: float
    passed: bool


class NumericConsistencyScorer(weave.Scorer):
    """Scorer that checks if numeric values in output appear in the context."""

    threshold: float = 0.7

    @weave.op
    async def score(self, context: str, output: str) -> NumericScore:  # type: ignore
        """
        Compare numeric tokens in the model output with those in the provided context.

        Robust to:
        - Scientific notation (e.g., 1.23e6).
        - Magnitude suffixes (k, M/mm, B/bn, T; words, singular or plural).
        - Currency symbols and thousands separators.
        - Parentheses negatives and varying decimals.
        - Date/time substrings are ignored.

        Args:
            context: The source text against which numeric consistency is checked.
            output: The assistant/model output being validated.

        Returns:
            A dictionary containing the fraction of matching numbers and pass/fail status.
        """
        output_numbers = _numbers(output)
        if not output_numbers:
            return {"fraction": 1.0, "passed": True}

        context_numbers = _numbers(context) if context else []
        fraction = _fraction_matched(output_numbers, context_numbers)
        return {"fraction": fraction, "passed": fraction >= self.threshold}


# Instantiate the custom Numeric Consistency Scorer
numeric_scorer = NumericConsistencyScorer()


@output_guardrail
async def numeric_consistency_guardrail(ctx: Any, agent: Any, agent_output: str) -> GuardrailFunctionOutput:
    """
    Guardrail that checks numeric consistency between the agent output and the retrieved context.

    The retrieved context is provided by the tool outputs.
    This guardrail uses our custom `NumericConsistencyScorer`, a sub-class of `Weave.Scorer`.

    Args:
        ctx: Run context wrapper providing access to `MyRunContext`.
        agent: The agent instance (unused).
        agent_output: The agent's textual output to validate.

    Returns:
        GuardrailFunctionOutput indicating numeric consistency results.
    """
    context = ctx.context
    context_text = f"{context.user_question}\n{context.tool_outputs}"
    score = await numeric_scorer.score(context=context_text, output=agent_output)
    fraction = float(score["fraction"])
    passed = bool(score["passed"])
    return GuardrailFunctionOutput(
        output_info={"numeric_consistency_fraction": fraction},
        tripwire_triggered=not passed,
    )


async def answer_questions(user_query: str, model: Optional[str] = None) -> Tuple[str, AgentHooksBase]:
    """Execute user questions through the Weave YFinance Agent.

    Args:
        user_query: The user query to answer.
        model: If provided, overrides the default model for the agent.

    Returns:
        The answer to the user query.
        If the response is blocked by a guardrail, a placeholder message is returned.
        The agent hooks are also returned.
    """
    server_path = os.path.join(os.getcwd(), "yfinance_server", "tools_yfinance.py")
    if not os.path.isfile(server_path):
        logger.error("Server script not found: %s", server_path)
        raise FileNotFoundError("tools_yfinance.py not found in the working directory.")
    logger.info("Found MCP server script at: %s", server_path)

    if model is None:
        model = MODEL

    @weave.op(name="Agent Workflow")
    async def _run(agent: Agent, user_query: str, ctx: MyRunContext) -> Tuple[str, Call]:
        """
        Run the agent for a given user question and context.

        Args:
            agent: The agent to run.
            user_query: The user question to answer.
            ctx: The context to use for the run.

        Returns:
            The answer to the user query.
        """
        call = weave.require_current_call()
        try:
            result = await Runner.run(starting_agent=agent, input=user_query, context=ctx)
            answer = result.final_output
            # Add Weave reaction to the call
            call.feedback.add_reaction("✅")
        except OutputGuardrailTripwireTriggered:
            logger.warning("Output guardrail triggered for query: '%s'.", user_query)
            answer = GUARDRAIL_MESSAGE
            # Add Weave reaction to the call
            call.feedback.add_reaction("❌")
        return answer, call

    async with MCPServerStdio(
        name="YFinance MCP Server",
        params={"command": shutil.which("python") or "python", "args": [server_path]},
    ) as mcp_server:
        logger.info("Started MCP server via stdio transport")

        agent = Agent(
            name="FinancialAssistant",
            instructions=instructions_financial_assistant,
            mcp_servers=[mcp_server],
            model=model,
            output_guardrails=[hallucination_guardrail, numeric_consistency_guardrail],
        )
        logger.info("Agent initialized with model: %s", model)
        logger.info("\n**User:** %s", user_query)
        ctx = MyRunContext(user_question=user_query)
        agent.hooks = CaptureToolOutputs()

        # Run the agent and get the answer and the call
        answer, call = await _run(agent, user_query, ctx)

        logger.info("**Agent:** %s\n", answer)

    return answer, agent.hooks


if __name__ == "__main__":
    # Example queries to test the agent:
    user_question = "What were the highest and lowest prices of AAPL in the last 6 months?"
    # "What were the highest and lowest prices of AAPL in the last 6 months?"
    # "Can you plot the stock price history of AAPL over the past year?"
    # "What are Apple's total assets and liabilities in its latest balance sheet?"
    # "How much revenue and net income did Apple report last year?"
    # "How much cash did Apple generate from operating activities in the most recent year?"
    # "Give me a summary of Apple's key financial figures for the last year (revenue, profit, etc.)."
    # "Who are the major shareholders of Apple and what percentage of the company do they hold?"
    # "Which institutions hold the most Apple stock, and how many shares do they own?"
    # "What are Apple's environmental, social, and governance (ESG) scores and ratings?"
    # "Tell me about Apple Inc. (sector, market cap, number of employees, etc.)."
    # # This question will trigger the guardrails
    # "Predict the highest and lowest prices of AAPL for the next 6 months?"
    try:
        asyncio.run(answer_questions(user_question))
    except Exception as e:
        logger.exception("Failed to run example queries: %s", e)
