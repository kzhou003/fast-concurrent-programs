"""
PROBLEM: Implement a Simple Agent System

Build an agent system where an LLM-based agent can use tools, maintain memory,
and execute multi-step tasks. This requires managing state, tool calling,
and thought/action/observation loops.

REQUIREMENTS:
- Agent maintains conversation history (context)
- Support tool definitions and tool calling
- Implement ReAct pattern (Reasoning, Acting, Observing)
- Track agent state (thinking, tool-using, done)
- Handle tool execution and result integration
- Support max steps to prevent infinite loops
- Proper error handling and fallback

PERFORMANCE NOTES:
- Should support 1000+ token context efficiently
- Tool execution should be fast (non-blocking)
- State management should be low-overhead

TEST CASE EXPECTATIONS:
- Agent should call tools appropriately
- Tool results should be integrated into context
- Agent should reach terminal states
- Multi-step reasoning should work
- Should handle missing or invalid tools gracefully
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json
import re


class AgentState(Enum):
    """States an agent can be in."""

    THINKING = "thinking"
    USING_TOOL = "using_tool"
    DONE = "done"
    ERROR = "error"


@dataclass
class Tool:
    """Represents a tool the agent can use."""

    name: str
    description: str
    parameters: Dict[str, str]  # parameter_name -> type description
    func: Callable

    def __call__(self, **kwargs) -> str:
        """Execute the tool."""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "user", "assistant", "tool"
    content: str


@dataclass
class AgentStep:
    """Represents one step of agent execution."""

    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    state: AgentState = AgentState.THINKING


class SimpleAgent:
    """A simple ReAct-based agent."""

    def __init__(
        self,
        model_fn: Callable,  # Function that takes messages and returns text
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
        memory_size: int = 20,
    ):
        """
        Initialize agent.

        Args:
            model_fn: Function that generates responses from conversation
            tools: List of available tools
            max_steps: Maximum number of steps before terminating
            memory_size: Maximum conversation history to keep
        """
        self.model_fn = model_fn
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.memory_size = memory_size

        self.conversation_history: List[Message] = []
        self.step_count = 0
        self.state = AgentState.THINKING

    def add_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def _build_context(self) -> str:
        """Build context string from conversation history."""
        # TODO: Build formatted conversation context
        context = ""
        for msg in self.conversation_history[-self.memory_size :]:
            context += f"{msg.role.upper()}: {msg.content}\n"
        return context

    def _parse_response(self, response: str) -> AgentStep:
        """
        Parse model response into structured step.

        Expected format:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: <json>

        or final answer:
        Final Answer: <answer>
        """
        # TODO: Parse response to extract thought, action, action_input
        # Return AgentStep with parsed information
        step = AgentStep()

        # Check for final answer
        if "Final Answer:" in response:
            step.state = AgentState.DONE
            step.thought = response.split("Final Answer:")[1].strip()
            return step

        # Try to parse structured format
        lines = response.split("\n")
        for line in lines:
            if line.startswith("Thought:"):
                step.thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
                step.action = action
                step.state = AgentState.USING_TOOL
            elif line.startswith("Action Input:"):
                try:
                    action_input = json.loads(line.replace("Action Input:", "").strip())
                    step.action_input = action_input
                except json.JSONDecodeError:
                    pass

        return step

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """
        Execute a tool.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input parameters for tool

        Returns:
            Result of tool execution
        """
        # TODO: Find tool and execute it
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        tool = self.tools[tool_name]
        return tool(**tool_input)

    def step(self) -> AgentStep:
        """Execute one agent step."""
        # TODO: Implement one agent reasoning step
        # 1. Build context from history
        # 2. Call model to get next action
        # 3. Parse response
        # 4. If tool action, execute tool
        # 5. Add to history
        # 6. Return step

        if self.step_count >= self.max_steps:
            self.state = AgentState.DONE
            return AgentStep(state=AgentState.DONE, thought="Max steps reached")

        self.step_count += 1

        # Get context
        context = self._build_context()

        # TODO: Call model
        response = self.model_fn(context)
        self.conversation_history.append(Message(role="assistant", content=response))

        # Parse response
        step = self._parse_response(response)

        # If tool action, execute
        if step.state == AgentState.USING_TOOL and step.action and step.action_input:
            result = self._execute_tool(step.action, step.action_input)
            step.observation = result
            self.conversation_history.append(Message(role="tool", content=result))
        else:
            self.state = AgentState.DONE

        return step

    def run(self, task: str, max_steps: Optional[int] = None) -> str:
        """
        Run agent on a task.

        Args:
            task: Task description
            max_steps: Optional override for max steps

        Returns:
            Final answer from agent
        """
        if max_steps:
            self.max_steps = max_steps

        # Reset state
        self.conversation_history = [Message(role="user", content=task)]
        self.step_count = 0
        self.state = AgentState.THINKING

        # TODO: Run agent loop until done
        final_answer = ""

        while self.state != AgentState.DONE and self.step_count < self.max_steps:
            step = self.step()
            if step.state == AgentState.DONE:
                final_answer = step.thought or ""
                break

        return final_answer

    def get_history(self) -> List[Message]:
        """Return conversation history."""
        return self.conversation_history


# Example tools
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return eval(expression)
    except:
        return 0.0


def search(query: str) -> str:
    """Simulate web search."""
    # Simulated search results
    results = {
        "weather": "It is sunny and 72°F",
        "python": "Python is a programming language",
        "machine learning": "Machine learning is a subset of AI",
    }
    return results.get(query.lower(), "No results found")


def test_basic_tool_execution():
    """Test that agent can execute tools."""

    def mock_model(context):
        # Simulate model output
        return """Thought: I should calculate 2 + 2
Action: calculator
Action Input: {"expression": "2 + 2"}"""

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[Tool("calculator", "Calculate math", {}, calculator)],
        max_steps=1,
    )

    step = agent.step()

    assert step.state == AgentState.USING_TOOL
    assert step.action == "calculator"
    assert step.observation == "4"

    print(f"✓ Basic tool execution test passed")


def test_tool_parsing():
    """Test parsing of model responses."""

    def mock_model(context):
        return """Thought: I need to search for information
Action: search
Action Input: {"query": "python"}"""

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[Tool("search", "Search", {}, search)],
    )

    step = agent.step()

    assert step.action == "search"
    assert step.action_input == {"query": "python"}

    print(f"✓ Tool parsing test passed")


def test_multi_step_execution():
    """Test multi-step agent execution."""
    step_counter = [0]

    def mock_model(context):
        step_counter[0] += 1
        if step_counter[0] == 1:
            return """Thought: First, I'll calculate 10 * 5
Action: calculator
Action Input: {"expression": "10 * 5"}"""
        elif step_counter[0] == 2:
            return """Thought: Now I'll add 50 + 25
Action: calculator
Action Input: {"expression": "50 + 25"}"""
        else:
            return "Final Answer: The result is 75"

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[Tool("calculator", "Calculate", {}, calculator)],
        max_steps=3,
    )

    step1 = agent.step()
    assert step1.action == "calculator"

    step2 = agent.step()
    assert step2.action == "calculator"

    step3 = agent.step()
    assert step3.state == AgentState.DONE

    print(f"✓ Multi-step execution test passed")


def test_final_answer():
    """Test agent reaching final answer."""

    def mock_model(context):
        return "Final Answer: The answer is 42"

    agent = SimpleAgent(model_fn=mock_model, max_steps=1)

    step = agent.step()

    assert step.state == AgentState.DONE
    assert "42" in step.thought

    print(f"✓ Final answer test passed")


def test_max_steps_limit():
    """Test that agent respects max steps."""

    def mock_model(context):
        return """Thought: Repeat
Action: calculator
Action Input: {"expression": "1+1"}"""

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[Tool("calculator", "Calc", {}, calculator)],
        max_steps=3,
    )

    result = agent.run("Calculate something")

    assert agent.step_count <= 3

    print(f"✓ Max steps limit test passed")


def test_memory_management():
    """Test that agent maintains conversation history."""

    def mock_model(context):
        if len(context.split("\n")) < 10:
            return """Thought: Continue
Action: calculator
Action Input: {"expression": "1+1"}"""
        else:
            return "Final Answer: Done"

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[Tool("calculator", "Calc", {}, calculator)],
        memory_size=5,
        max_steps=10,
    )

    agent.run("Test task")

    # Memory should not exceed size
    assert len(agent.conversation_history) <= 10 * 2  # Approximate

    print(f"✓ Memory management test passed")


def test_tool_not_found():
    """Test handling of invalid tool calls."""

    def mock_model(context):
        return """Thought: Use nonexistent tool
Action: nonexistent_tool
Action Input: {}"""

    agent = SimpleAgent(model_fn=mock_model, max_steps=1)

    step = agent.step()

    # Should handle error gracefully
    assert step.observation is not None
    assert "Error" in step.observation or "not found" in step.observation

    print(f"✓ Tool not found test passed")


def test_conversation_history():
    """Test that conversation history is maintained."""

    def mock_model(context):
        return "Final Answer: Test complete"

    agent = SimpleAgent(model_fn=mock_model)

    agent.run("Initial task")

    history = agent.get_history()

    assert len(history) >= 2  # At least user and assistant
    assert history[0].role == "user"

    print(f"✓ Conversation history test passed")


def test_multiple_tools():
    """Test agent with multiple available tools."""

    def mock_model(context):
        if "add" in context:
            return """Action: calculator
Action Input: {"expression": "2+3"}"""
        else:
            return """Action: search
Action Input: {"query": "test"}"""

    agent = SimpleAgent(
        model_fn=mock_model,
        tools=[
            Tool("calculator", "Math", {}, calculator),
            Tool("search", "Search", {}, search),
        ],
        max_steps=1,
    )

    step = agent.step()

    assert step.action in ["calculator", "search"]

    print(f"✓ Multiple tools test passed")


if __name__ == "__main__":
    print("Running Agent System tests...\n")

    test_basic_tool_execution()
    test_tool_parsing()
    test_multi_step_execution()
    test_final_answer()
    test_max_steps_limit()
    test_memory_management()
    test_tool_not_found()
    test_conversation_history()
    test_multiple_tools()

    print("\n✓ All tests passed!")
