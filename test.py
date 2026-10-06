import asyncio
from my_mcp.mcp_client import get_client


async def main():
    # 1. Initialize client
    client = get_client()

    # 2. Await tools for the 'weather' server
    weather_tools = await client.get_tools(server_name="weather")

    # 3. Print or work with the tools
    print(f"Loaded {len(weather_tools)} weather tools:")
    for tool in weather_tools:
        print(f" - {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())