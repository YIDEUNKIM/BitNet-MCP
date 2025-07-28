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
    # 입력된 location 값을 소문자로 변환하여 비교합니다.
    if location.lower() == "seoul":
        return {"location": "seoul", "temperature": 25, "condition": "clean"} # 'clean'을 '맑음'으로 수정
    else:
        # 일치하지 않는 지역에 대한 오류 메시지를 더 명확하게 반환합니다.
        return {"location": location, "error": "error"}

@mcp.tool()
def get_user_info(username: str):
    """Return the information of specified user."""
    if username == "yideun":
        return {"username": "yideun", "status": "active", "email": "yideun@example.com"}
    else:
        return {"username": username, "error": "유저를 찾을 수 없습니다."}

if __name__ == "__main__":
    print("mcp server 시작")
    mcp.run(transport="sse", host="localhost", port=8001)
