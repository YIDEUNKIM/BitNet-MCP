# mcp_server.py
from fastmcp import FastMCP
import random

# FastMCP 서버 인스턴스를 생성합니다.
mcp = FastMCP(name="MyToolServer")

@mcp.tool
def get_weather(city: str) -> str:
    """
    지정된 도시의 날씨 정보를 반환합니다. (예시를 위해 무작위 날씨를 생성합니다)
    """
    weathers = ["맑음", "흐림", "비", "눈", "천둥번개"]
    temperature = random.randint(-5, 35)
    print(f"'{city}' 날씨 요청 처리 중...")
    return f"도시 '{city}'의 현재 날씨는 '{random.choice(weathers)}'이며, 온도는 {temperature}°C 입니다."

@mcp.tool
def calculate(a: int, b: int, operation: str = "add") -> str:
    """
    두 숫자에 대해 지정된 연산(add, subtract, multiply, divide)을 수행합니다.
    """
    print(f"계산 요청 처리 중: {a} {operation} {b}")
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "오류: 0으로 나눌 수 없습니다."
        result = a / b
    else:
        return f"오류: 지원하지 않는 연산 '{operation}' 입니다."
    return f"연산 결과: {result}"

if __name__ == "__main__":
    print("MCP 서버를 시작합니다. 주소: http://localhost:8000/mcp")
    # SSE (Server-Sent Events) 전송 방식을 사용하여 서버를 실행합니다.
    mcp.run(transport="sse", host="localhost", port=8000)
