#!/usr/bin/env python3
"""
Data Analysis Workflow Example

This script demonstrates using a local LLM as a data analysis assistant.
It shows how to structure prompts for exploratory data analysis, statistical interpretation,
and visualization suggestions.
"""

import os
import sys
import argparse
import subprocess
import tempfile

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.chat_history import create_chat_session

# Define ANSI colors for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def run_llm_query(
    model_path, 
    prompt, 
    framework="llama.cpp", 
    max_tokens=1024, 
    temperature=0.2
):
    """Run a query against a local LLM using the specified framework."""
    if framework == "llama.cpp":
        # Create a temporary file for the prompt
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(prompt)
            prompt_file = temp_file.name
        
        try:
            # Determine llama.cpp path
            llamacpp_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../llama.cpp-setup/build/main'
            ))
            
            # Run llama.cpp
            cmd = [
                llamacpp_path,
                '-m', model_path,
                '-n', str(max_tokens),
                '--temp', str(temperature),
                '-f', prompt_file,
                '--color'
            ]
            
            # Add Metal acceleration if on macOS
            if sys.platform == "darwin":
                cmd.append('--metal')
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print(f"{COLORS['red']}Error running llama.cpp:{COLORS['reset']}")
                print(result.stderr)
                return ""
            
            # Extract the assistant's response based on the template
            output = result.stdout
            try:
                output = output.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
            except IndexError:
                pass
                
            return output
            
        finally:
            # Clean up the temporary file
            os.unlink(prompt_file)
            
    elif framework == "mlx":
        try:
            # Import MLX modules
            import mlx.core as mx
            from mlx_lm import load, generate
            
            # Load the model
            model, tokenizer = load(model_path)
            
            # Format as a list of messages for MLX
            messages = []
            lines = prompt.split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                if "<|im_start|>system" in line:
                    current_role = "system"
                    current_content = []
                elif "<|im_start|>user" in line:
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "user"
                    current_content = []
                elif "<|im_start|>assistant" in line:
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "assistant"
                    current_content = []
                elif "<|im_end|>" in line:
                    continue
                else:
                    current_content.append(line)
            
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            
            # Construct the prompt
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Fallback to simple concatenation
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n\n"
                prompt += "Assistant: "
            
            # Generate text
            tokens = generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens, 
                temp=temperature
            )
            
            return tokenizer.decode(tokens)
            
        except ImportError:
            print(f"{COLORS['red']}Error: MLX is not installed. Install with: pip install mlx mlx-lm{COLORS['reset']}")
            return ""
    
    return ""


def exploratory_data_analysis_example(model_path, framework="llama.cpp"):
    """Example workflow for exploratory data analysis with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Exploratory Data Analysis Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for data analysis
    chat = create_chat_session(
        system_message=(
            "You are an expert data scientist specializing in exploratory data analysis. "
            "You provide clear, step-by-step guidance on analyzing datasets, identifying patterns, "
            "and extracting insights. You suggest appropriate statistical techniques and visualizations "
            "based on the data characteristics. Your advice is practical and implementation-focused."
        )
    )
    
    # Sample dataset description
    dataset_description = """
Dataset: Customer Purchase History
Size: 50,000 records
Time Period: January 2020 - December 2022

Columns:
- customer_id: Unique identifier for each customer
- purchase_date: Date of purchase (YYYY-MM-DD)
- product_category: Category of product purchased (Electronics, Clothing, Home Goods, Books, Groceries, etc.)
- product_id: Unique identifier for each product
- amount: Purchase amount in USD
- payment_method: Method of payment (Credit Card, PayPal, Bank Transfer, etc.)
- customer_age: Age of customer at time of purchase
- customer_gender: Gender of customer (M/F/Other)
- customer_location: City, State format
- is_discount_applied: Boolean indicating if a discount was applied
- discount_amount: Amount of discount applied (0 if no discount)
- is_first_purchase: Boolean indicating if this is the customer's first purchase
    """
    
    # Add user query for EDA suggestions
    chat.add_user_message(
        f"I have a new dataset of customer purchase history that I need to analyze. "
        f"I'd like to perform an exploratory data analysis to understand spending patterns "
        f"and customer behavior. Here's the dataset description:\n\n{dataset_description}\n\n"
        f"What initial EDA steps should I take? What kind of insights should I look for?"
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating EDA suggestions...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.3  # Slightly higher temperature for creative analysis suggestions
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Exploratory Data Analysis Suggestions:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Follow-up question about specific analysis
    chat.add_user_message(
        "I'm particularly interested in understanding seasonal purchasing patterns and how they "
        "differ across product categories. Can you suggest specific analyses and visualizations "
        "for this?"
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating seasonal analysis suggestions...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.3
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Seasonal Analysis Suggestions:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def statistical_analysis_example(model_path, framework="llama.cpp"):
    """Example workflow for statistical analysis guidance with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Statistical Analysis Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for statistical analysis
    chat = create_chat_session(
        system_message=(
            "You are an expert statistician specializing in applied statistics for data science. "
            "You provide clear explanations of statistical concepts and practical guidance on "
            "selecting and interpreting statistical tests. You are thorough in your analysis "
            "and careful to mention assumptions and limitations. You provide Python code examples "
            "when appropriate."
        )
    )
    
    # Add user query about statistical test selection
    chat.add_user_message(
        "I'm analyzing a marketing campaign dataset. I want to determine if there's a significant "
        "difference in conversion rates between three different email subject lines we tested. "
        "Each subject line was sent to approximately 5,000 customers, and I have the binary conversion "
        "data (converted: yes/no) for each customer. What statistical test should I use, and how "
        "should I interpret the results?"
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating statistical test recommendations...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.2  # Lower temperature for precise statistical guidance
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Statistical Test Recommendation:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Add a follow-up question with sample results
    chat.add_user_message(
        "I ran the Chi-square test as you suggested, and here are the results:\n\n"
        "Subject Line A: 520 conversions out of 5,000 (10.4%)\n"
        "Subject Line B: 578 conversions out of 5,000 (11.6%)\n"
        "Subject Line C: 492 conversions out of 5,000 (9.8%)\n\n"
        "Chi-square statistic: 8.76\n"
        "p-value: 0.0125\n\n"
        "How should I interpret these results? Also, should I run any post-hoc tests to determine "
        "which subject lines are significantly different from each other?"
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating results interpretation...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.2
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Statistical Results Interpretation:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def data_visualization_example(model_path, framework="llama.cpp"):
    """Example workflow for data visualization guidance with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Data Visualization Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for data visualization
    chat = create_chat_session(
        system_message=(
            "You are an expert in data visualization and communication of insights. "
            "You provide practical guidance on selecting the most appropriate visualization types "
            "for different data and storytelling goals. You emphasize best practices in visualization "
            "design, including accessibility, clarity, and honesty in representation. "
            "You provide Python code examples using matplotlib, seaborn, or plotly when appropriate."
        )
    )
    
    # Add user query about visualizing a complex dataset
    chat.add_user_message(
        "I'm working with a climate dataset that includes the following variables for 200 global cities "
        "over a 50-year period (1970-2020):\n\n"
        "- Average monthly temperature (°C)\n"
        "- Monthly precipitation (mm)\n"
        "- Annual extreme weather events count\n"
        "- Population density\n"
        "- Green space percentage\n"
        "- Air quality index (AQI)\n\n"
        "I want to create compelling visualizations that show how climate patterns have changed over time "
        "and how they correlate with the other variables. What visualization approaches would you recommend? "
        "I'm particularly interested in showing the relationships between multiple variables simultaneously."
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating visualization recommendations...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.3  # Slightly higher temperature for creative visualization ideas
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Visualization Recommendations:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Add a follow-up question about code implementation
    chat.add_user_message(
        "I'd like to implement your suggestion for an interactive dashboard using Plotly. "
        "Can you provide a code example for creating a dashboard that includes:\n\n"
        "1. A heatmap showing temperature changes over time\n"
        "2. A scatter plot with selectable variables for the axes\n"
        "3. A choropleth map showing geographic distribution of one variable\n\n"
        "I'm comfortable with Python and have the data in a pandas DataFrame called 'climate_df'."
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating visualization code...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=2048,  # More tokens for code example
        temperature=0.2   # Lower temperature for precise code
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Visualization Code Example:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def main():
    """Main function to run the data analysis workflow examples."""
    parser = argparse.ArgumentParser(description="Data Analysis Workflow Example")
    parser.add_argument(
        "--model", "-m", 
        type=str,
        help="Path to the model file or directory"
    )
    parser.add_argument(
        "--framework", "-f", 
        type=str, 
        choices=["llama.cpp", "mlx"],
        default="llama.cpp",
        help="Framework to use (llama.cpp or mlx)"
    )
    parser.add_argument(
        "--example", "-e", 
        type=str, 
        choices=["eda", "stats", "viz", "all"],
        default="all",
        help="Which example to run (eda=Exploratory Data Analysis, stats=Statistical Analysis, viz=Visualization)"
    )
    
    args = parser.parse_args()
    
    # If no model specified, look for available models
    if not args.model:
        if args.framework == "llama.cpp":
            models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../llama.cpp-setup/models'
            ))
            models = [
                os.path.join(models_dir, f) 
                for f in os.listdir(models_dir) 
                if f.endswith('.gguf')
            ] if os.path.exists(models_dir) else []
        else:  # MLX
            models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../mlx-setup/models'
            ))
            models = [
                os.path.join(models_dir, d) 
                for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d)) and 
                os.path.exists(os.path.join(models_dir, d, 'config.json'))
            ] if os.path.exists(models_dir) else []
        
        if not models:
            print(f"{COLORS['red']}No models found. Please specify a model with --model.{COLORS['reset']}")
            return 1
        
        # Use the first model found
        args.model = models[0]
        print(f"{COLORS['yellow']}Using model: {args.model}{COLORS['reset']}")
    
    # Run the selected example(s)
    if args.example in ["eda", "all"]:
        exploratory_data_analysis_example(args.model, args.framework)
    
    if args.example in ["stats", "all"]:
        statistical_analysis_example(args.model, args.framework)
    
    if args.example in ["viz", "all"]:
        data_visualization_example(args.model, args.framework)
    
    print(f"{COLORS['bold']}{COLORS['green']}All examples completed!{COLORS['reset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())