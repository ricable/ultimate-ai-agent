# TinyDM    

Author: [Michael Hightower](https://github.com/Cmhight03/TinyDM)

A lightweight, chat-based Dungeon Master assistant that helps DMs create and manage D&D 5E sessions through natural conversation. TinyDM integrates with Live Agent Studio to provide creative content generation, rules assistance, and improvisational support.  (Be kind - this is my "Hello World")

## Features

- 🎲 Quick encounter generation
- 🎭 NPC creation and personality development
- 🏰 Dynamic location descriptions
- 📚 D&D 5E rules clarification
- ⚔️ Tactical combat suggestions
- 🌟 Creative story elements and plot hooks

## Setup Guide

1. **Prerequisites**
   - Python 3.11+
   - PowerShell (Windows)
   - Supabase account (create one at https://supabase.com)
   - Google Cloud account with Gemini API access (set up at https://cloud.google.com)

2. **Installation**
   ```powershell
   # Clone the repository
   git clone https://github.com/Cmhight03/TinyDM.git
   cd TinyDM

   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\Activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configuration**
   Create a `.env` file in the project root with your credentials:
   ```env
   # API Configuration
   HOST=localhost
   PORT=8001
   BEARER_TOKEN=your_secure_token_here

   # Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here

   # Supabase Configuration
   SUPABASE_URL=your_project.supabase.co
   SUPABASE_KEY=your_anon_key
   ```

   Required steps:
   1. Create a Supabase project and copy your project URL and anon key
   2. Set up a Google Cloud project and enable the Gemini API
   3. Generate a secure bearer token for API authentication

4. **Starting the Server**
   ```powershell
   # Ensure virtual environment is activated
   .\venv\Scripts\Activate

   # Start the FastAPI server
   uvicorn main:app --host localhost --port 8001 --reload
   ```

   The server will be available at `http://localhost:8001`

## User Guide

### Getting Started

TinyDM uses natural language processing to understand and respond to your needs. Simply describe what you want in conversational terms.

### Core Commands

1. **Encounter Generation**
   ```
   "Create an encounter for 4 level 3 players in a forest"
   "I need a random dungeon battle"
   "Generate a hard combat for level 5 players"
   ```

2. **Location Description**
   ```
   "Describe a bustling tavern"
   "Create a creepy dungeon entrance"
   "Help me design a wizard's tower"
   ```

3. **NPC Creation**
   ```
   "Create a mysterious merchant NPC"
   "Generate a town guard captain"
   "I need a villain for my campaign"
   ```

4. **Rules Help**
   ```
   "How does grappling work?"
   "Explain opportunity attacks"
   "What are the rules for darkness?"
   ```

### Sample Prompts

1. **Quick Combat Setup**
   ```
   User: "I need a quick forest encounter for 4 level 2 players"
   TinyDM: "Let me create a balanced forest encounter for your party...
   
   A pack of wolves has been stalking the party. You have:
   - 3 wolves (CR 1/4 each)
   - 1 dire wolf pack leader (CR 1)
   
   Tactical Setup:
   - Wolves approach from different directions
   - Pack leader waits to flank weak targets
   - Dense undergrowth provides partial cover
   
   Would you like the stat blocks or tactical suggestions?"
   ```

2. **Creative Location**
   ```
   User: "Describe a mysterious shrine in the woods"
   TinyDM: "Deep in the forest stands an ancient shrine, its stone walls wrapped in luminescent vines. 
   
   Key features:
   - Crumbling stone archway covered in unknown symbols
   - Small reflecting pool with crystal-clear water
   - Stone altar with mysterious offerings
   - Soft, ethereal humming in the air
   
   Would you like more details about the symbols, the offerings, or potential plot hooks?"
   ```

3. **Quick NPC**
   ```
   User: "Create a suspicious merchant"
   TinyDM: "Meet Darian Blacktrade:
   
   Physical: Well-dressed halfling with a silver-capped cane and nervous eyes
   Personality: Overly friendly but constantly glancing over his shoulder
   Hook: Claims to sell rare items at suspiciously low prices
   
   Would you like his inventory, mannerisms, or potential plot hooks?"
   ```

**Credit URL**: https://github.com/Cmhight03/TinyDM with honorable mention for https://www.dumbdumbdice.com/dumb-dumbs-dragons 

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
