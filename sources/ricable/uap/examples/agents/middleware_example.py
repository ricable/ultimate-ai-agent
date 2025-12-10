"""
Middleware Example

This example demonstrates how to add middleware to agents for request/response processing.
"""

import asyncio
import json
from typing import Dict, Any, Tuple
from datetime import datetime
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration


async def logging_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Middleware that logs all messages."""
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] MIDDLEWARE LOG: User message: {message}")
    
    # Add logging metadata to context
    context['logged_at'] = timestamp
    context['message_length'] = len(message)
    
    return message, context


async def profanity_filter_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Middleware that filters profanity (demo implementation)."""
    
    # Simple profanity filter (in production, use a proper library)
    blocked_words = ['badword1', 'badword2', 'inappropriate']
    
    filtered_message = message
    for word in blocked_words:
        if word in message.lower():
            filtered_message = filtered_message.replace(word, "*" * len(word))
            context['profanity_filtered'] = True
    
    return filtered_message, context


async def translation_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Middleware that detects and translates simple phrases."""
    
    # Simple translation dictionary (in production, use a translation service)
    translations = {
        'hola': 'hello',
        'bonjour': 'hello', 
        'guten tag': 'hello',
        'ciao': 'hello',
        'gracias': 'thank you',
        'merci': 'thank you',
        'danke': 'thank you',
        'arigato': 'thank you'
    }
    
    original_message = message
    translated_message = message.lower()
    
    for foreign, english in translations.items():
        if foreign in translated_message:
            translated_message = translated_message.replace(foreign, english)
            context['translated'] = True
            context['original_language'] = 'detected'
    
    # Capitalize properly
    if context.get('translated'):
        translated_message = translated_message.capitalize()
        print(f"TRANSLATION MIDDLEWARE: '{original_message}' -> '{translated_message}'")
    
    return translated_message if context.get('translated') else message, context


async def sentiment_analysis_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Middleware that adds sentiment analysis to context."""
    
    # Simple sentiment analysis (in production, use ML models)
    positive_words = ['happy', 'good', 'great', 'excellent', 'wonderful', 'love', 'like', 'amazing']
    negative_words = ['sad', 'bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible']
    
    message_lower = message.lower()
    
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
        confidence = positive_count / (positive_count + negative_count + 1)
    elif negative_count > positive_count:
        sentiment = 'negative'
        confidence = negative_count / (positive_count + negative_count + 1)
    else:
        sentiment = 'neutral'
        confidence = 0.5
    
    context['sentiment'] = {
        'classification': sentiment,
        'confidence': confidence,
        'positive_indicators': positive_count,
        'negative_indicators': negative_count
    }
    
    return message, context


async def response_enhancer_middleware(response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Middleware that enhances responses with additional metadata."""
    
    if 'metadata' not in response:
        response['metadata'] = {}
    
    # Add processing information
    response['metadata']['processed_by_middleware'] = True
    response['metadata']['processing_time'] = datetime.utcnow().isoformat()
    
    # Add sentiment-based response modification
    if 'sentiment' in context:
        sentiment = context['sentiment']
        response['metadata']['user_sentiment'] = sentiment['classification']
        
        # Modify response based on sentiment
        if sentiment['classification'] == 'negative' and sentiment['confidence'] > 0.7:
            content = response.get('content', '')
            response['content'] = f"I understand you might be feeling frustrated. {content}"
        elif sentiment['classification'] == 'positive' and sentiment['confidence'] > 0.7:
            content = response.get('content', '')
            response['content'] = f"{content} I'm glad you're in a good mood! 😊"
    
    # Add translation acknowledgment
    if context.get('translated'):
        content = response.get('content', '')
        response['content'] = f"{content}\n\n(Note: I detected you might be using another language and translated your message)"
    
    return response


class MiddlewareDemo:
    """Demo class showing different middleware configurations."""
    
    @staticmethod
    async def basic_middleware_demo():
        """Demonstrate basic middleware functionality."""
        print("=== Basic Middleware Demo ===\n")
        
        config = Configuration()
        
        # Create agent with logging middleware
        agent = (CustomAgentBuilder("middleware-demo")
                .with_simple_framework()
                .with_config(config)
                .add_middleware(logging_middleware)
                .build())
        
        await agent.start()
        
        messages = [
            "Hello there!",
            "How are you today?",
            "Can you help me with something?"
        ]
        
        for message in messages:
            print(f"User: {message}")
            response = await agent.process_message(message)
            print(f"Agent: {response['content']}")
            print(f"Metadata: {response.get('metadata', {})}")
            print()
        
        await agent.stop()
    
    @staticmethod
    async def multi_middleware_demo():
        """Demonstrate multiple middleware working together."""
        print("=== Multi-Middleware Demo ===\n")
        
        config = Configuration()
        
        # Create agent with multiple middleware layers
        agent = (CustomAgentBuilder("multi-middleware-demo")
                .with_simple_framework()
                .with_config(config)
                .add_middleware(logging_middleware)
                .add_middleware(profanity_filter_middleware)
                .add_middleware(translation_middleware)
                .add_middleware(sentiment_analysis_middleware)
                .build())
        
        # Add response middleware
        async def enhanced_process_message(message: str, context: Dict = None):
            """Wrapper to add response middleware."""
            response = await agent.process_message(message, context)
            # Apply response enhancement
            enhanced_response = await response_enhancer_middleware(response, context or {})
            return enhanced_response
        
        await agent.start()
        
        test_messages = [
            "Hola! How are you?",
            "I'm feeling really happy today!",
            "This is terrible and I hate it",
            "Bonjour, can you help me?",
            "You're amazing and wonderful!",
            "Gracias for your help"
        ]
        
        for message in test_messages:
            print(f"User: {message}")
            response = await enhanced_process_message(message)
            print(f"Agent: {response['content']}")
            
            # Show middleware processing results
            metadata = response.get('metadata', {})
            if 'user_sentiment' in metadata:
                print(f"Detected sentiment: {metadata['user_sentiment']}")
            print()
        
        await agent.stop()
    
    @staticmethod
    async def custom_middleware_demo():
        """Demonstrate creating custom middleware."""
        print("=== Custom Middleware Demo ===\n")
        
        # Custom middleware that tracks conversation history
        conversation_history = []
        
        async def conversation_tracker_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            """Track conversation turns."""
            turn_number = len(conversation_history) + 1
            
            conversation_history.append({
                'turn': turn_number,
                'user_message': message,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            context['conversation_turn'] = turn_number
            context['conversation_history'] = conversation_history[-3:]  # Last 3 turns
            
            print(f"CONVERSATION TRACKER: Turn {turn_number}")
            return message, context
        
        # Custom middleware that adds time-based greetings
        async def time_based_greeting_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            """Add time-based context to greetings."""
            hour = datetime.now().hour
            
            if any(greeting in message.lower() for greeting in ['hello', 'hi', 'hey']):
                if hour < 12:
                    time_context = "Good morning"
                elif hour < 17:
                    time_context = "Good afternoon"
                else:
                    time_context = "Good evening"
                
                context['time_greeting'] = time_context
                print(f"TIME GREETING: Added context '{time_context}'")
            
            return message, context
        
        config = Configuration()
        
        agent = (CustomAgentBuilder("custom-middleware-demo")
                .with_simple_framework()
                .with_config(config)
                .add_middleware(conversation_tracker_middleware)
                .add_middleware(time_based_greeting_middleware)
                .add_middleware(sentiment_analysis_middleware)
                .build())
        
        await agent.start()
        
        messages = [
            "Hello!",
            "How's the weather?",
            "I'm having a great day!",
            "Can you help me?",
            "Thanks, goodbye!"
        ]
        
        for message in messages:
            print(f"User: {message}")
            response = await agent.process_message(message)
            print(f"Agent: {response['content']}")
            
            # Show conversation tracking
            metadata = response.get('metadata', {})
            if 'conversation_turn' in response.get('metadata', {}):
                print(f"Conversation turn: {metadata.get('conversation_turn')}")
            
            print()
        
        print("Final conversation history:")
        for turn in conversation_history:
            print(f"  Turn {turn['turn']}: {turn['user_message']}")
        
        await agent.stop()


async def main():
    """Main function demonstrating middleware examples."""
    print("=== UAP Middleware Examples ===\n")
    
    demo = MiddlewareDemo()
    
    await demo.basic_middleware_demo()
    print("\n" + "="*60 + "\n")
    
    await demo.multi_middleware_demo()
    print("\n" + "="*60 + "\n")
    
    await demo.custom_middleware_demo()


if __name__ == "__main__":
    asyncio.run(main())