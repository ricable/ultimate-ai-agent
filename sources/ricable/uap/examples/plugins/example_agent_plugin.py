"""
Example Agent Plugin

This plugin demonstrates how to create agent plugins that can handle specific message types.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from uap_sdk.plugin import AgentPlugin, Configuration


class CalculatorPlugin(AgentPlugin):
    """Plugin that provides calculator functionality to agents."""
    
    PLUGIN_NAME = "calculator"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides calculator functionality for mathematical operations"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
        self.calculation_history = []
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the calculator plugin."""
        print(f"Calculator plugin initialized")
        self.calculation_history = []
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"Calculator plugin cleaned up. Performed {len(self.calculation_history)} calculations.")
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this plugin should handle the message."""
        message_lower = message.lower()
        
        # Handle if message contains math operators or keywords
        math_indicators = ['+', '-', '*', '/', '^', '=', 'calculate', 'math', 'sum', 'multiply', 'divide']
        return any(indicator in message_lower for indicator in math_indicators)
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process mathematical expressions and queries."""
        
        if not self.should_handle_message(message, context):
            return None
        
        message_lower = message.lower()
        
        try:
            # Handle different types of math queries
            if 'history' in message_lower:
                return await self._get_calculation_history()
            
            elif 'clear' in message_lower and 'history' in message_lower:
                return await self._clear_history()
            
            elif any(op in message for op in ['+', '-', '*', '/', '^', '**']):
                return await self._perform_calculation(message, agent_id)
            
            elif 'sqrt' in message_lower or 'square root' in message_lower:
                return await self._calculate_square_root(message)
            
            elif 'factorial' in message_lower:
                return await self._calculate_factorial(message)
            
            elif 'percentage' in message_lower or '%' in message:
                return await self._calculate_percentage(message)
            
            else:
                return {
                    "content": "I can help with:\n• Basic math (2+2, 10*5, etc.)\n• Square roots (sqrt 16)\n• Factorials (factorial 5)\n• Percentages (20% of 100)\n• History (show calculation history)",
                    "metadata": {
                        "plugin": self.name,
                        "type": "help",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        except Exception as e:
            return {
                "content": f"Calculator error: {str(e)}",
                "metadata": {
                    "plugin": self.name,
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def _perform_calculation(self, expression: str, agent_id: str) -> Dict[str, Any]:
        """Perform basic arithmetic calculation."""
        # Clean up the expression
        cleaned_expr = expression.replace('^', '**')  # Python exponentiation
        
        # Safety check - only allow numbers and basic operators
        import re
        if re.match(r'^[\d\+\-\*\/\(\)\.\s\*\*]+$', cleaned_expr):
            try:
                result = eval(cleaned_expr)
                
                # Store in history
                calculation = {
                    'expression': expression,
                    'result': result,
                    'agent_id': agent_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.calculation_history.append(calculation)
                
                return {
                    "content": f"{expression} = {result}",
                    "metadata": {
                        "plugin": self.name,
                        "type": "calculation",
                        "expression": expression,
                        "result": result,
                        "calculation_count": len(self.calculation_history),
                        "timestamp": calculation['timestamp']
                    }
                }
            except Exception as e:
                return {
                    "content": f"Calculation error: {str(e)}",
                    "metadata": {
                        "plugin": self.name,
                        "type": "calculation_error",
                        "error": str(e)
                    }
                }
        else:
            return {
                "content": "For safety, I can only evaluate basic math expressions with numbers and operators (+, -, *, /, ^)",
                "metadata": {
                    "plugin": self.name,
                    "type": "safety_error"
                }
            }
    
    async def _calculate_square_root(self, message: str) -> Dict[str, Any]:
        """Calculate square root of a number."""
        import re
        import math
        
        numbers = re.findall(r'\d+\.?\d*', message)
        if numbers:
            num = float(numbers[0])
            if num < 0:
                return {
                    "content": "Cannot calculate square root of negative number",
                    "metadata": {"plugin": self.name, "type": "domain_error"}
                }
            
            result = math.sqrt(num)
            
            # Store in history
            calculation = {
                'expression': f'sqrt({num})',
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.calculation_history.append(calculation)
            
            return {
                "content": f"√{num} = {result:.6f}",
                "metadata": {
                    "plugin": self.name,
                    "type": "sqrt",
                    "input": num,
                    "result": result,
                    "timestamp": calculation['timestamp']
                }
            }
        else:
            return {
                "content": "Please provide a number for square root calculation (e.g., 'sqrt 16')",
                "metadata": {"plugin": self.name, "type": "input_error"}
            }
    
    async def _calculate_factorial(self, message: str) -> Dict[str, Any]:
        """Calculate factorial of a number."""
        import re
        import math
        
        numbers = re.findall(r'\d+', message)
        if numbers:
            num = int(numbers[0])
            if num < 0:
                return {
                    "content": "Cannot calculate factorial of negative number",
                    "metadata": {"plugin": self.name, "type": "domain_error"}
                }
            elif num > 170:  # Prevent extremely large calculations
                return {
                    "content": "Number too large for factorial calculation (max: 170)",
                    "metadata": {"plugin": self.name, "type": "overflow_error"}
                }
            
            result = math.factorial(num)
            
            # Store in history
            calculation = {
                'expression': f'{num}!',
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.calculation_history.append(calculation)
            
            return {
                "content": f"{num}! = {result}",
                "metadata": {
                    "plugin": self.name,
                    "type": "factorial",
                    "input": num,
                    "result": result,
                    "timestamp": calculation['timestamp']
                }
            }
        else:
            return {
                "content": "Please provide a number for factorial calculation (e.g., 'factorial 5')",
                "metadata": {"plugin": self.name, "type": "input_error"}
            }
    
    async def _calculate_percentage(self, message: str) -> Dict[str, Any]:
        """Calculate percentage."""
        import re
        
        # Look for patterns like "20% of 100" or "what is 20% of 100"
        percentage_match = re.search(r'(\d+\.?\d*)%\s*of\s*(\d+\.?\d*)', message)
        if percentage_match:
            percentage = float(percentage_match.group(1))
            total = float(percentage_match.group(2))
            result = (percentage / 100) * total
            
            # Store in history
            calculation = {
                'expression': f'{percentage}% of {total}',
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.calculation_history.append(calculation)
            
            return {
                "content": f"{percentage}% of {total} = {result}",
                "metadata": {
                    "plugin": self.name,
                    "type": "percentage",
                    "percentage": percentage,
                    "total": total,
                    "result": result,
                    "timestamp": calculation['timestamp']
                }
            }
        else:
            return {
                "content": "Please use format like '20% of 100' for percentage calculations",
                "metadata": {"plugin": self.name, "type": "format_error"}
            }
    
    async def _get_calculation_history(self) -> Dict[str, Any]:
        """Get calculation history."""
        if not self.calculation_history:
            return {
                "content": "No calculations performed yet.",
                "metadata": {"plugin": self.name, "type": "history", "count": 0}
            }
        
        recent_calculations = self.calculation_history[-10:]  # Last 10
        history_text = "Recent calculations:\n"
        for i, calc in enumerate(recent_calculations, 1):
            history_text += f"{i}. {calc['expression']} = {calc['result']}\n"
        
        return {
            "content": history_text,
            "metadata": {
                "plugin": self.name,
                "type": "history",
                "total_calculations": len(self.calculation_history),
                "shown": len(recent_calculations)
            }
        }
    
    async def _clear_history(self) -> Dict[str, Any]:
        """Clear calculation history."""
        count = len(self.calculation_history)
        self.calculation_history = []
        
        return {
            "content": f"Cleared {count} calculations from history.",
            "metadata": {
                "plugin": self.name,
                "type": "history_cleared",
                "cleared_count": count
            }
        }


class TimePlugin(AgentPlugin):
    """Plugin that provides time and date information."""
    
    PLUGIN_NAME = "time"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides current time, date, and timezone information"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the time plugin."""
        print("Time plugin initialized")
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print("Time plugin cleaned up")
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this plugin should handle the message."""
        time_keywords = ['time', 'date', 'clock', 'now', 'today', 'timezone', 'utc']
        return any(keyword in message.lower() for keyword in time_keywords)
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process time-related queries."""
        
        if not self.should_handle_message(message, context):
            return None
        
        message_lower = message.lower()
        
        try:
            from datetime import datetime, timezone
            import time
            
            now = datetime.now()
            utc_now = datetime.utcnow()
            
            if 'utc' in message_lower:
                response = f"Current UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            elif 'date' in message_lower:
                response = f"Today's date: {now.strftime('%Y-%m-%d (%A)')}"
            elif 'timezone' in message_lower:
                response = f"Local timezone: {time.tzname[0]} (UTC{time.timezone//3600:+d})"
            else:
                # Default: current local time
                response = f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return {
                "content": response,
                "metadata": {
                    "plugin": self.name,
                    "type": "time_info",
                    "local_time": now.isoformat(),
                    "utc_time": utc_now.isoformat(),
                    "timestamp": utc_now.isoformat()
                }
            }
        
        except Exception as e:
            return {
                "content": f"Time plugin error: {str(e)}",
                "metadata": {
                    "plugin": self.name,
                    "type": "error",
                    "error": str(e)
                }
            }


class QuotePlugin(AgentPlugin):
    """Plugin that provides inspirational quotes."""
    
    PLUGIN_NAME = "quotes"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Provides inspirational quotes and wisdom"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
        self.quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Innovation distinguishes between a leader and a follower. - Steve Jobs",
            "Life is what happens to you while you're busy making other plans. - John Lennon",
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
            "It is during our darkest moments that we must focus to see the light. - Aristotle",
            "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
            "The only impossible journey is the one you never begin. - Tony Robbins",
            "In the middle of difficulty lies opportunity. - Albert Einstein"
        ]
        self.quote_index = 0
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the quotes plugin."""
        print(f"Quotes plugin initialized with {len(self.quotes)} quotes")
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        print("Quotes plugin cleaned up")
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this plugin should handle the message."""
        quote_keywords = ['quote', 'inspiration', 'motivate', 'wisdom', 'inspire']
        return any(keyword in message.lower() for keyword in quote_keywords)
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process quote requests."""
        
        if not self.should_handle_message(message, context):
            return None
        
        import random
        
        # Get a random quote
        quote = random.choice(self.quotes)
        self.quote_index = (self.quote_index + 1) % len(self.quotes)
        
        return {
            "content": f"Here's an inspiring quote for you:\n\n{quote}",
            "metadata": {
                "plugin": self.name,
                "type": "quote",
                "quote": quote,
                "total_quotes": len(self.quotes),
                "timestamp": datetime.utcnow().isoformat()
            }
        }


# Example usage and testing
async def test_plugins():
    """Test the example plugins."""
    from uap_sdk import PluginManager, Configuration, UAPAgent, CustomAgentBuilder
    
    print("=== Testing Agent Plugins ===\n")
    
    # Create configuration and plugin manager
    config = Configuration()
    plugin_manager = PluginManager(config)
    
    # Manually load plugins (simulating plugin discovery)
    calc_plugin = CalculatorPlugin()
    time_plugin = TimePlugin()
    quote_plugin = QuotePlugin()
    
    await calc_plugin.enable(config)
    await time_plugin.enable(config)
    await quote_plugin.enable(config)
    
    print("Plugins enabled successfully\n")
    
    # Create an agent
    agent = (CustomAgentBuilder("plugin-demo")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    await agent.start()
    
    # Test messages that should trigger different plugins
    test_messages = [
        "What's 5 + 3?",  # Calculator
        "What time is it?",  # Time
        "I need some inspiration",  # Quotes
        "Calculate 10 * 6",  # Calculator
        "Show me the date",  # Time
        "sqrt 25",  # Calculator
        "Give me a motivating quote",  # Quotes
        "What's 20% of 150?",  # Calculator
        "Show calculation history",  # Calculator
        "What's the UTC time?",  # Time
        "Hello there!"  # Default agent response
    ]
    
    for message in test_messages:
        print(f"User: {message}")
        
        # Try each plugin to see if it handles the message
        handled = False
        
        for plugin in [calc_plugin, time_plugin, quote_plugin]:
            if plugin.should_handle_message(message, {}):
                response = await plugin.process_message("plugin-demo", message, {})
                if response:
                    print(f"Plugin ({plugin.name}): {response['content']}")
                    print(f"Metadata: {response.get('metadata', {})}")
                    handled = True
                    break
        
        if not handled:
            # Use default agent
            response = await agent.process_message(message)
            print(f"Agent: {response['content']}")
        
        print()
    
    # Cleanup
    await calc_plugin.disable()
    await time_plugin.disable()
    await quote_plugin.disable()
    await agent.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_plugins())