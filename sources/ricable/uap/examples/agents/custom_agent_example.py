"""
Custom Agent Example

This example shows how to create a custom agent framework with specific capabilities.
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from uap_sdk import UAPAgent, AgentFramework, Configuration


class WeatherAgent(AgentFramework):
    """A custom agent that provides weather information."""
    
    def __init__(self, config: Configuration = None):
        super().__init__("weather", config)
        self.weather_data = {
            "new york": {"temp": 22, "condition": "sunny", "humidity": 65},
            "london": {"temp": 15, "condition": "cloudy", "humidity": 80},
            "tokyo": {"temp": 28, "condition": "rainy", "humidity": 85},
            "paris": {"temp": 18, "condition": "partly cloudy", "humidity": 70},
            "sydney": {"temp": 25, "condition": "sunny", "humidity": 60}
        }
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather-related queries."""
        message_lower = message.lower()
        
        # Greeting
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return {
                "content": "Hello! I'm your weather agent. Ask me about the weather in any city!",
                "metadata": {"type": "greeting", "timestamp": datetime.utcnow().isoformat()}
            }
        
        # Weather query
        elif "weather" in message_lower:
            # Extract city from message
            city = None
            for city_name in self.weather_data.keys():
                if city_name in message_lower:
                    city = city_name
                    break
            
            if city:
                weather = self.weather_data[city]
                response = (f"Weather in {city.title()}:\n"
                           f"Temperature: {weather['temp']}°C\n"
                           f"Condition: {weather['condition']}\n"
                           f"Humidity: {weather['humidity']}%")
            else:
                available_cities = ", ".join(self.weather_data.keys())
                response = f"I can provide weather for: {available_cities}. Which city would you like to know about?"
            
            return {
                "content": response,
                "metadata": {
                    "type": "weather_info",
                    "city": city,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # List cities
        elif any(word in message_lower for word in ["cities", "locations", "where"]):
            cities = list(self.weather_data.keys())
            response = f"I have weather data for {len(cities)} cities: {', '.join(cities)}"
            
            return {
                "content": response,
                "metadata": {"type": "city_list", "cities": cities}
            }
        
        # Default response
        else:
            return {
                "content": "I'm a weather agent. Ask me about the weather in any city, or say 'cities' to see available locations.",
                "metadata": {"type": "help", "timestamp": datetime.utcnow().isoformat()}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get weather agent status."""
        return {
            "status": self.status,
            "framework": self.framework_name,
            "initialized": self.is_initialized,
            "available_cities": len(self.weather_data),
            "capabilities": self.get_capabilities()
        }
    
    async def initialize(self) -> None:
        """Initialize the weather agent."""
        self.is_initialized = True
        self.status = "active"
        print("Weather agent initialized with data for", len(self.weather_data), "cities")
    
    def get_capabilities(self) -> List[str]:
        """Get weather agent capabilities."""
        return ["weather_queries", "city_information", "temperature_data", "conditions"]


class MathAgent(AgentFramework):
    """A custom agent that performs mathematical calculations."""
    
    def __init__(self, config: Configuration = None):
        super().__init__("math", config)
        self.operation_count = 0
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process mathematical queries."""
        message_lower = message.lower()
        
        # Greeting
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return {
                "content": "Hello! I'm your math agent. I can help with calculations, equations, and math problems!",
                "metadata": {"type": "greeting", "timestamp": datetime.utcnow().isoformat()}
            }
        
        # Math operations
        try:
            # Simple calculator functionality
            if any(op in message for op in ["+", "-", "*", "/", "^", "**"]):
                # Replace ^ with ** for Python exponentiation
                expression = message.replace("^", "**")
                
                # Basic safety check - only allow numbers and basic operators
                import re
                if re.match(r'^[\d\+\-\*\/\(\)\.\s\*\*]+$', expression):
                    try:
                        result = eval(expression)
                        self.operation_count += 1
                        
                        return {
                            "content": f"{message} = {result}",
                            "metadata": {
                                "type": "calculation",
                                "expression": message,
                                "result": result,
                                "operation_count": self.operation_count,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    except Exception as e:
                        return {
                            "content": f"Sorry, I couldn't calculate that: {str(e)}",
                            "metadata": {"type": "error", "error": str(e)}
                        }
                else:
                    return {
                        "content": "For safety, I can only evaluate basic math expressions with numbers and operators (+, -, *, /, ^)",
                        "metadata": {"type": "safety_error"}
                    }
            
            # Square root
            elif "sqrt" in message_lower or "square root" in message_lower:
                import re
                numbers = re.findall(r'\d+', message)
                if numbers:
                    import math
                    num = float(numbers[0])
                    result = math.sqrt(num)
                    self.operation_count += 1
                    
                    return {
                        "content": f"√{num} = {result:.4f}",
                        "metadata": {
                            "type": "sqrt",
                            "input": num,
                            "result": result,
                            "operation_count": self.operation_count
                        }
                    }
            
            # Powers of numbers
            elif "power" in message_lower or "^" in message:
                import re
                numbers = re.findall(r'\d+', message)
                if len(numbers) >= 2:
                    base = float(numbers[0])
                    exp = float(numbers[1])
                    result = base ** exp
                    self.operation_count += 1
                    
                    return {
                        "content": f"{base}^{exp} = {result}",
                        "metadata": {
                            "type": "power",
                            "base": base,
                            "exponent": exp,
                            "result": result,
                            "operation_count": self.operation_count
                        }
                    }
            
            # Statistics
            elif "stats" in message_lower or "statistics" in message_lower:
                return {
                    "content": f"Math operations performed: {self.operation_count}",
                    "metadata": {"type": "statistics", "operation_count": self.operation_count}
                }
            
        except Exception as e:
            return {
                "content": f"Math error: {str(e)}",
                "metadata": {"type": "error", "error": str(e)}
            }
        
        # Default math help
        return {
            "content": "I can help with:\n• Basic calculations (2+2, 10*5, etc.)\n• Square roots (sqrt 16)\n• Powers (2^8)\n• Statistics (stats)",
            "metadata": {"type": "help", "timestamp": datetime.utcnow().isoformat()}
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get math agent status."""
        return {
            "status": self.status,
            "framework": self.framework_name,
            "initialized": self.is_initialized,
            "operations_performed": self.operation_count,
            "capabilities": self.get_capabilities()
        }
    
    async def initialize(self) -> None:
        """Initialize the math agent."""
        self.is_initialized = True
        self.status = "active"
        print("Math agent initialized")
    
    def get_capabilities(self) -> List[str]:
        """Get math agent capabilities."""
        return ["arithmetic", "square_root", "exponentiation", "statistics"]


async def demo_weather_agent():
    """Demonstrate the weather agent."""
    print("=== Weather Agent Demo ===\n")
    
    config = Configuration()
    weather_framework = WeatherAgent(config)
    weather_agent = UAPAgent("weather-demo", weather_framework, config)
    
    await weather_agent.start()
    
    weather_queries = [
        "Hello!",
        "What's the weather like in London?",
        "How about New York?",
        "Which cities do you have weather for?",
        "Tell me about Tokyo weather",
        "What's the temperature in Paris?"
    ]
    
    for query in weather_queries:
        print(f"User: {query}")
        response = await weather_agent.process_message(query)
        print(f"Weather Agent: {response['content']}")
        print()
    
    await weather_agent.stop()


async def demo_math_agent():
    """Demonstrate the math agent."""
    print("=== Math Agent Demo ===\n")
    
    config = Configuration()
    math_framework = MathAgent(config)
    math_agent = UAPAgent("math-demo", math_framework, config)
    
    await math_agent.start()
    
    math_queries = [
        "Hi there!",
        "2 + 3",
        "10 * 5",
        "100 / 4",
        "2^8",
        "sqrt 16",
        "What's 15 + 27 * 2?",
        "stats",
        "Can you help with calculus?"
    ]
    
    for query in math_queries:
        print(f"User: {query}")
        response = await math_agent.process_message(query)
        print(f"Math Agent: {response['content']}")
        print()
    
    await math_agent.stop()


async def main():
    """Main function demonstrating custom agents."""
    print("=== UAP Custom Agent Examples ===\n")
    
    await demo_weather_agent()
    print("\n" + "="*50 + "\n")
    await demo_math_agent()


if __name__ == "__main__":
    asyncio.run(main())