from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_time():
    """Return the current time."""
    import datetime
    return {"current_time": str(datetime.datetime.now())}

@mcp.tool()
def get_weather(location: str):
    """Returns weather information for a specific area."""
    # Convert the input location value to lowercase for comparison.
    if location.lower() == "seoul":
        return {"location": "seoul", "temperature": 25, "condition": "clean"}
    else:
        # Returns a more specific error message for non-matching locations.
        return {"location": location, "error": "error"}

@mcp.tool()
def get_user_info(username: str):
    """Return the information of specified user."""
    if username == "yideun":
        return {"username": "yideun", "status": "active", "email": "yideun@example.com"}
    else:
        return {"username": username, "error": "User not found."}

if __name__ == "__main__":
    print("Starting mcp server")
    mcp.run(transport="sse", host="localhost", port=8001)

