# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments, kernel_function

"""
The following sample demonstrates how to create a chat completion agent that
answers questions about a sample menu using a Semantic Kernel Plugin. The Chat
Completion Service is first added to the kernel, and the kernel is passed in to the
ChatCompletionAgent constructor. Additionally, the plugin is supplied via the kernel.
To enable auto-function calling, the prompt execution settings are retrieved from the kernel
using the specified `service_id`. The function choice behavior is set to `Auto` to allow the
agent to automatically execute the plugin's functions when needed.
"""



# Simulate a conversation with the agent
USER_INPUTS = [
"Plan me a day trip.",
"I don't like that destination. Plan me another vacation.",
]

import random   

# Define a sample plugin for the sample

class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        # List of vacation destinations
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        # Track last destination to avoid repeats
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        # Get available destinations (excluding last one if possible)
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        # Select a random destination
        destination = random.choice(available_destinations)

        # Update the last destination
        self.last_destination = destination

        return destination

AGENT_NAME = "TravelAgent"
AGENT_INSTRUCTIONS = "You are a helpful AI Agent that can help plan vacations for customers at random destinations"

async def main():
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "agent"
    kernel = Kernel()
    kernel.add_plugin(DestinationsPlugin(), plugin_name="destinations")
    kernel.add_service(AzureChatCompletion(service_id=service_id))

    # 2. Configure the function choice behavior to auto invoke kernel functions
    # so that the agent can automatically execute the menu plugin functions when needed
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 3. Create the agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name=AGENT_NAME ,
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(settings=settings),
    )

    # 4. Create a thread to hold the conversation
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: ChatHistoryAgentThread = None

    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 5. Invoke the agent for a response
        async for response in agent.invoke(messages=user_input, thread=thread):
            print(f"# {response.name}: {response}")
            thread = response.thread

    # 6. Cleanup: Clear the thread
    await thread.delete() if thread else None

 


if __name__ == "__main__":
    asyncio.run(main())
