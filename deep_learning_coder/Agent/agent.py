import ollama
import re

# Define the Agent class
class SimpleAgent:
    def __init__(self, model_name):
        self.model_name = model_name

    def use_tool(self, tool_name, *args):
        # A simple dispatcher to call the right tool
        if tool_name == "addition":
            return self.addition_tool(*args)
        else:
            return "Tool not found"

    # The tool function that performs addition
    def addition_tool(self, a, b):
        return a + b

    # The method where the agent reacts to inputs
    def react(self, input_text):
        # Call the Ollama model directly
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": input_text}])
        return response['text']

    # The agent's main logic to perform addition or other tasks
    def perform_task(self, task_description):
        # Check if the task is to add numbers
        if "add" in task_description.lower():
            # Extract numbers from the input string (use regex to find all integers)
            numbers = [int(s) for s in re.findall(r'-?\d+', task_description)]
            if len(numbers) == 2:
                # Use the tool to add the numbers
                result = self.use_tool("addition", numbers[0], numbers[1])
                return f"The result of adding {numbers[0]} and {numbers[1]} is {result}."
            else:
                return "Could not find exactly two numbers to add."
        else:
            return "Unknown task."

# Initialize the agent with the DeepSeek-Coder v2 model
agent = SimpleAgent("deepseek-coder-v2")

# Example task input for addition
task = "Please add 3 and 7."

# The agent reacts to the task and provides the result
result = agent.perform_task(task)
print(result)

