import os
import asyncio
import certifi

from .mcp_client import get_client

import logging

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


async def tavily_mcp_search(user_query: str):

    # getting the MCP Client
    client = get_client()
    if client == None:
        print("No MCP client found.")
        return 
    
    # getting all available tools from the mcp server
    tools = await client.get_tools(server_name="tavily")

    # checking if we have the tavily search tool
    tavily_search_tool = next((tool for tool in tools if tool.name=='tavily_search'), None)

    # returning if not tool found
    if tavily_search_tool == None:
        print("No such tool exists!")
        return

    # getting the result from the search tool after passing the user query
    result = await tavily_search_tool.ainvoke(
        {
            "query": user_query
        }
    )

    if not result:
        print("No result found from the tool!")
        return 

    return result


async def aviationstack_mcp(
        tool_name: str,
        tool_args: dict = None
):

    # getting the MCP Client
    client = get_client()
    if client == None:
        print("No MCP client found.")
        return 
    
    # getting all available tools from the mcp server
    aviationstack_tools = await client.get_tools(server_name="aviationstack")

    # getting the tool_name
    tool_to_use = next((tool for tool in aviationstack_tools if tool.name==tool_name), None)
    
    # returning if no tool found
    if not tool_to_use:
        print("No such tool exists!")
        return

    # getting the result from the search tool after passing the user query
    result = await tool_to_use.ainvoke(
        tool_args or {}
    )

    if not result:
        print("No result found from the tool!")
        return 

    return result


async def weather_mcp_search(city: str, meteorology: str):

    # 1. Getting the MCP Client
    client = get_client()
    if client is None:
        print("No MCP client found.")
        return 

    # 2. Get the list of tools (remove curly braces)
    weather_tools = await client.get_tools(server_name="weather")

    # 3. Get the city_weather tool safely
    city_weather_tool = next(
        (tool for tool in weather_tools if tool.name == "get_current_city_weather"), 
        None
    )

    # 4. Get the city_forecast tool safely
    city_forecast_tool = next(
        (tool for tool in weather_tools if tool.name == "get_city_forecast"), 
        None
    )

    # 5. Invoke the selected tool and return its result directly
    if meteorology == "weather":
        if not city_weather_tool:
            raise ValueError("Tool 'get_current_city_weather' not found.")
        return await city_weather_tool.ainvoke({"city": city})
    else:
        if not city_forecast_tool:
            raise ValueError("Tool 'get_city_forecast' not found.")
        return await city_forecast_tool.ainvoke({"city": city})

