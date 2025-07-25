from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_time():
    """현재 시간을 반환합니다."""
    import datetime
    return {"current_time": str(datetime.datetime.now())}

@mcp.tool()
def get_weather(location: str):
    """특정 지역의 날씨 정보를 반환합니다."""
    if location == "서울":
        return {"location": "서울", "temperature": 25, "condition": "맑음"}
    else:
        return {"location": location, "error": "날씨 정보를 찾을 수 없습니다."}

@mcp.tool()
def get_user_info(username: str):
    """특정 유저의 정보를 반환합니다."""
    if username == "yideun":
        return {"username": "yideun", "status": "active", "email": "yideun@example.com"}
    else:
        return {"username": username, "error": "유저를 찾을 수 없습니다."}

if __name__ == "__main__":
    print("mcp server 시작")
    mcp.run(transport="sse", host="localhost", port=8001)
