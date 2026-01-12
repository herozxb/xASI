import ollama
import re

# Define the Agent class
class SimpleAgent:
    def __init__(self, model_name):
        self.model_name = model_name

    # The tool function that performs addition
    def addition_tool(self, a, b):
        return a + b

    # The tool function that performs subtraction
    def subtraction_tool(self, a, b):
        return a - b

    # The tool function that performs multiplication
    def multiplication_tool(self, a, b):
        return a * b

    # The tool function that performs division
    def division_tool(self, a, b):
        if b != 0:
            return a / b
        else:
            return "Error: Division by zero"

    # The tool function that performs multiplication
    def or_tool(self, a, b):
        return a | b

    # The tool function that performs multiplication
    def search_tool(self, text):
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": text}])
        return response

    # The method where the agent reacts to inputs and uses the tool when necessary
    def react(self, input_text):
        # Define the agent's prompt to use the tools or other reasoning tasks
        prompt = f"""
        You are a coding assistant. You will reason through tasks and use the appropriate tools when necessary.

        # Tool Functions:
        addition_tool(a, b)  # This function takes two arguments and returns their sum.
        subtraction_tool(a, b)  # This function takes two arguments and returns their difference.
        multiplication_tool(a, b)  # This function takes two arguments and returns their product.
        division_tool(a, b)  # This function takes two arguments and returns their division (handling zero division).
        or_tool(a, b)  # This function takes two arguments and returns their division.
        search_tool(text)  # This function takes two arguments and returns their division.

        # Example tasks:
        Add 5 and 7.
        Subtract 5 from 7.
        Multiply 5 and 7.
        Divide 10 by 2.
        or 1 by 1
        search how to make an agent

        Task:
        {input_text}

        Please reason through the task and use the tool if necessary. Provide the result.
        """

        # Call the Ollama model directly for reasoning
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        
        # Extract the response content to find if the model asks to perform any tool operation
        response_content = response['message']['content']
        print(f"Response from model: {response_content}")

        # Look for pattern in the response to see if it asks to perform any operation
        match_add = re.search(r'addition_tool\((\d+), (\d+)\)', response_content)
        match_sub = re.search(r'subtraction_tool\((\d+), (\d+)\)', response_content)
        match_mul = re.search(r'multiplication_tool\((\d+), (\d+)\)', response_content)
        match_div = re.search(r'division_tool\((\d+), (\d+)\)', response_content)
        match_or = re.search(r'or_tool\((\d+), (\d+)\)', response_content)
        match_search = re.search(r'search_tool\s*\(\s*"([^"]+)"\s*\)', response_content)



        if match_add:
            print("===============use_addition_tools[0]==================")
            a = int(match_add.group(1))
            b = int(match_add.group(2))
            result = self.addition_tool(a, b)
            return result

        elif match_sub:
            print("===============use_substraction_tools[1]==================")
            a = int(match_sub.group(1))
            b = int(match_sub.group(2))
            result = self.subtraction_tool(a, b)
            return result

        elif match_mul:
            print("===============use_multiplication_tools[2]==================")
            a = int(match_mul.group(1))
            b = int(match_mul.group(2))
            result = self.multiplication_tool(a, b)
            return result

        elif match_div:
            print("===============use_division_tools[3]==================")
            a = int(match_div.group(1))
            b = int(match_div.group(2))
            result = self.division_tool(a, b)
            return result
        elif match_or:
            print("===============use_or_tools[4]==================")
            a = int(match_or.group(1))
            b = int(match_or.group(2))
            result = self.or_tool(a, b)
            return result
        elif match_search:
            print("===============use_or_tools[5]==================")
            text = match_search.group(1)
            result = self.search_tool(text)
            print("=====text_result====")
            print(text)
            print("=====search_result====")
            print(result['message']['content'])
            return result['message']['content']

        # If the model doesn't specifically call any tool, return the reasoning content
        return response_content

    # The agent's main logic to perform tasks
    def perform_task(self, task_description):
        # Here we simply send the task to the LLM to let it decide which tool to use
        result = self.react(task_description)
        return result


# Initialize the agent with the DeepSeek-Coder v2 model
agent = SimpleAgent("deepseek-coder-v2")

# Example task input for each operation
task_add = "Please add 3 and 7."
task_sub = "Please subtract 3 from 7."
task_mul = "Please multiply 3 and 7."
task_div = "Please divide 10 by 2."
task_or = "Please or 1 by 1."
task_search = "Please search how to make an agent."


# The agent reacts to each task and provides the result
result_add = agent.perform_task(task_add)
result_sub = agent.perform_task(task_sub)
result_mul = agent.perform_task(task_mul)
result_div = agent.perform_task(task_div)
result_or = agent.perform_task(task_or)
result_search = agent.perform_task(task_search)

# Print the results
print(f"Addition Result: {result_add}")
print(f"Subtraction Result: {result_sub}")
print(f"Multiplication Result: {result_mul}")
print(f"Division Result: {result_div}")
print(f"Or Result: {result_or}")
print(f"Search Result: {result_search}")
