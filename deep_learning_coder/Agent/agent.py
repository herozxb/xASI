import ollama
import re

# Define the Agent class
class SimpleAgent:
    def __init__(self, model_name):
        self.model_name = model_name

    # The tool function that performs addition
    def addition_tool(self, a, b):
        return a + b

    # The method where the agent reacts to inputs and uses the tool when necessary
    def react(self, input_text):
        # Define the agent's prompt to use the addition tool or other reasoning tasks
        prompt = f"""
        You are a coding assistant. You will reason through tasks and use the appropriate tools when necessary.

        # Tool Function:
        addition_tool(a, b)  # This function takes two arguments and returns their sum.

        # Example task:
        Add 5 and 7.

        Task:
        {input_text}

        Please reason through the task and use the tool if necessary. Provide the result.
        """

        # Call the Ollama model directly for reasoning
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        
        # Extracting the response content to find if the model mentions addition tool
        response_content = response['message']['content']
        
        print(f"Response from model: {response_content}")
        print("===========llm_response===========")
        
        # Look for pattern in the response to see if it asks to perform addition
        match = re.search(r'addition_tool\((\d+), (\d+)\)', response_content)
        if match:
            print("===========use_tools[0]===========")
            a = int(match.group(1))  # First number
            b = int(match.group(2))  # Second number
            result = self.addition_tool(a, b)  # Perform the actual addition
            
            return result  # Return the result of addition

        # If the model doesn't specifically call the addition tool, return the model's response
        return response_content

    # The agent's main logic to perform addition or other tasks
    def perform_task(self, task_description):
        # Here we simply send the task to the LLM to let it decide
        result = self.react(task_description)
        return result


# Initialize the agent with the DeepSeek-Coder v2 model
agent = SimpleAgent("deepseek-coder-v2")

# Example task input for addition
task = "Please add 3 and 7."

# The agent reacts to the task and provides the result
result = agent.perform_task(task)
print(f"Result: {result}")

