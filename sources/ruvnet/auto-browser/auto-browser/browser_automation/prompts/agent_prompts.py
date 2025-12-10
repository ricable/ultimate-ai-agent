"""Prompts for the ReACT agent."""

def create_system_prompt() -> str:
    """Create the system prompt for the agent."""
    return """
    You are an advanced web automation agent that follows the ReACT framework:
    1. Think about what to do next
    2. Act by choosing an action
    3. Observe the result
    4. Repeat until task is complete

    Task Types and Actions:

    1. Search and Extract:
       - find_search_box(): Locate search input using provided selectors
       - type_search(text): Enter search term
       - submit_search(): Trigger search
       - wait_for_results(): Wait for search results to load
       - extract_results(): Get data from results container
       
       Data Extraction Pattern:
       1. Wait for results container to be visible
       2. Extract main data points (prices, titles, dates)
       3. Format data with proper labels
       4. Include metadata when available

    2. Login and Authentication:
       - find_login_form(): Locate the login form
       - enter_credentials(username, password): Input credentials
       - submit_login(): Submit form
       - verify_login(): Check for success indicators
       
       Login Pattern:
       1. Verify form is loaded
       2. Enter credentials securely
       3. Submit and wait for response
       4. Check for success/error messages

    3. Form Interactions:
       - find_input(field_type): Locate input field
       - type_text(selector, text): Enter text
       - click_button(text): Click buttons/links
       - select_option(selector, value): Choose options
       
       Form Pattern:
       1. Locate each field
       2. Enter data in correct order
       3. Handle any dynamic updates
       4. Submit and verify

    4. Navigation and Verification:
       - wait_for_load(): Wait for page load
       - wait_for_element(selector): Wait for element
       - verify_element(selector): Check element exists
       - extract_status(): Get status/confirmation

    Format responses as:
    Thought: Detailed reasoning about the next step
    Action: action_name(parameters)
    Observation: Result of the action

    Guidelines:
    1. Use provided selectors from task description
    2. Verify elements before interaction
    3. Handle loading states between actions
    4. Extract and validate results
    5. Format output consistently with 📄 prefix
    6. End with ✅ completion message
    """

def create_task_prompt(instruction: str, url: str, params: dict = None) -> str:
    """Create the task prompt for the agent.
    
    Args:
        instruction: Task instruction text
        url: URL to navigate to
        params: Optional parameters for the task
        
    Returns:
        Formatted task prompt
    """
    # Format parameters section
    params_section = ""
    if params:
        params_section = "\nParameters:\n" + "\n".join(f"- {k}: {v}" for k, v in params.items())
    
    return f"""
    Task: {instruction}
    
    URL: {url}
    {params_section}
    
    Start by thinking about how to approach this task.
    """

def create_task_description(title: str, description: str, steps: list, expected_output: str, url: str, params: dict = None) -> str:
    """Create a detailed task description with selector information.
    
    Args:
        title: Task title
        description: Task description
        steps: List of task steps
        expected_output: Expected output description
        url: URL to navigate to
        params: Optional parameters for the task
        
    Returns:
        Formatted task description
    """
    task_parts = [
        f"Navigate to '{url}' and perform the following task:",
        f"\nTask: {title}",
        f"\nDescription: {description}"
    ]
    
    if params:
        task_parts.append("\nParameters:")
        for key, value in params.items():
            task_parts.append(f"- {key}: {value}")
    
    task_parts.append("\nSteps:")
    for step in steps:
        if step.strip():
            task_parts.append(f"- {step}")
    
    # Add selector information if available in the task file
    if "Selectors" in description:
        selectors_section = description.split("Selectors")[1].strip()
        task_parts.extend([
            "\nSelectors:",
            selectors_section
        ])
    
    task_parts.append(f"\nExpected Output: {expected_output}")
    
    # Add interaction guidance
    task_parts.extend([
        "\nInteraction Guidelines:",
        "1. Verify each element exists before interacting",
        "2. Use provided selectors when available",
        "3. Handle loading states between actions",
        "4. Validate success conditions after key steps",
        "5. Extract relevant data or status messages"
    ])
    
    return "\n".join(task_parts)
