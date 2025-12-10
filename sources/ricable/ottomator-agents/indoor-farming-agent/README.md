# INDOOR SMART FARMING AGENT

Author: [Aditya Prabhu](https://github.com/adityaprabhu16)

**Platform:** n8n (you can import the .json file into your own n8n instance to check out the flow)

**NOTE**: Most of this README is from the official repo for this agent which is why the images don't work here. Visit [the repo here](https://github.com/adityaprabhu16/GenerativeAIResearch/blob/master/IndoorFarmingAgent/README.md) to see the full README with images!

I've developed the following agent through n8n for the recent Ottomator hackathon.

Update: video finally uploaded - here's video demo of a couple of the unique features this agent can perform:
https://www.youtube.com/watch?v=aSW25hZ1XM0 

## Introduction:
For this Hackathon, my goal is to demo a robust Farming RAG agent that acts as a smart assistant for your indoor farm, greenhouse, or any other growing asset.

My aim here is to showcase the utility of agents in IoT applications, where data can change hourly and needs active monitoring to ensure the optimal conditions and the best health for the specific plants you are growing.


To showcase the utility of this agent, I set up two off the shelf wireless sensors that report hourly data from two grow spaces that I manage: the Coastal Greenhouse and the Tropical Greenhouse.

| Tropical Greenhouse | Coastal Greenhouse |
|---------------------|---------------------|
| ![Tropical Greenhouse](images/TropicalGreenhouse.JPG) | ![Coastal Greenhouse](images/CoastalGreenhouse.JPG) |


In addition to these two sensors, I've setup a small overhead IoT camera which the agent is able to return remote images from when asked, giving you a sneak peek at what the plants and greenhouse look like, without actually being there!
 ![IoT Camera](images/RemoteCamera.JPG)

The agent I designed helps you...
 - ...visualize data from live greenhouse sensors,
 - ...monitor changes to your greenhouse parameters,
 - ...adjust growing controls, from anywhere in the world!
 - ...learn more about the greenhouses and updated documentation
 - ...and more!

As the name suggests, the two greenhouses are meant to emulate slightly different climates in two distinct geographies, though in the winter the conditions happen to overlap slightly. Note the difference in both the soil type and general flora from the overhead view of the greenhouses above!

These IoT integration possibilities are endless, but for the scope of this Hackathon I'm focusing on environmental (humidity, temperature, air pressure, sensor battery) and visual (JPG image updates from camera)

## Greenhouses:
Here’s some more information about the sensors in each greenhouse that the indoor farming agent will help monitor.

### Tropical Greenhouse:

The Tropical Greenhouse is aimed to emulate the environmental conditions of approximately the Kibara Plateau, DRC. This climate is known to be highland tropical, with seasonal variations in rainfall and temperature. Currently, this climate will be a bit cold and dry, and flora that grow on this plateau will often go into an obligate dormancy. A well known species from this region, also critically endangered, is the Drosera Katangensis, a unique stem-forming insectivorous plant with light cream-colored dewy leaves.

### Coastal Greenhouse

The Coastal greenhouse is aimed to emulate the environmental conditions of approximately the coastal plains of Perth, Australia. This climate is known to have dry summers with little rainfall, and cooler wetter winters. Currently, the climate is a bit cold and wet, and flora that grow on these plains tend to emerge from their tubers to begin their active growing season with the harsh summers behind them. A well known species from this region is the Drosera Aberrans, a tuberous insectivorous plant with dark jade colored leaves arranged in a rosette. This plant often has a distinct sweet fragrance.

## Using the Agent! A Video Demo.
To be added.

## Using the Agent! A Written Demo.
The following is just a small subset of what you could ask, but my goal is to showcase each of the distinct capabilities here that a grower would find useful for their operation.


### 1. 🟩 Visualize Incoming Data from the Greenhouses! (Chart Agent Functionality)
![Demo1](images/Demo1.png)

### 2. 🟥 Set Thresholds that Trigger Alerts when out of bounds! (Discord / Flask Integration)!
![Demo2a](images/Demo2a.png)
| Phone Notification View | Discord Alert View |
|---------------------|---------------------|
| ![Demo2b](images/Demo2b.jpg) | ![Demo2c](images/Demo2c.jpg) |

I made the Discord publically available. You can actually join the Discord at https://discord.gg/wVcxdFPf28 - it should be available for up to 100 invitations.

### 3. 🟦 View Real Greenhouse Image Data! (Custom Camera Endpoint Integration)
![Demo1](images/Demo3.png)

### 4. 🟨 Live Document Integration: Ask Questions about the Greenhouse Documentation and get Updates! (Document Parsing, Semantic Search)
![Demo1](images/Demo4a.png)
![Demo1](images/Demo4b.png)

### 5. 🟪 Analyze Data and Look for Trends (Context Based Querying and Filtering)
![Demo1](images/Demo5a.png)
![Demo1](images/Demo5b.png)


Things you can ask the agent:
- For some ideas to ask the agent that a grower like me would be interested in, try the following:
- What is the current temperature and humidity of the tropical greenhouse?
- When did we last hear from our tropical greenhouse?
- Help me visualize.... (you can visualize temp, humidity, battery, air pressure!)
- Let's see what .... looks like currently, send me the latest snapshot! (only Coastal Greenhouse has image data available)
- It's winter time, so let's update the high temperature trigger to 25 degrees Celsius 
- - You can choose from temperature, humidity, (low, high or both), for both greenhouses (Coastal, Tropical)!
And so on!


Final Notes:

Charting and generating visuals is the most intensive query for the agent to process because I've had to fine-tine (through trial and error) the DB filtering prompts and output parser to prevent the agent from producing fake, malformed, and even nonexistent data! It can take around 40s-48s depending on how much you asked it to plot, but the meaningful visual at the end is worth it.
All other queries get satisfied within several seconds.

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.