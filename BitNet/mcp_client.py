import argparse
import subprocess
import os
import platform
import json
import asyncio
from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError, NotFoundError

def get_llama_cli_path():
    """
    운영체제에 맞는 llama-cli 실행 파일의 전체 경로를 구성합니다.
    """
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"llama-cli not found at {main_path}. Please build the project first.")
    return main_path

def query_local_llm(prompt: str, args) -> str:
    """
    llama-cli를 서브프로세스로 호출하여 로컬 LLM의 응답을 받아옵니다.
    """
    try:
        llama_cli_path = get_llama_cli_path()
        command = [
            f'{llama_cli_path}',
            '-m', args.model,
            '-p', prompt,
            '-n', str(args.n_predict),
            '-t', str(args.threads),
            '-c', str(args.ctx_size),
            '--temp', str(args.temperature),
            '--no-display-prompt' # 프롬프트 자체는 출력하지 않도록 옵션 추가
        ]
        # GPU 사용 옵션이 있다면 추가
        if hasattr(args, 'n_gpu_layers') and args.n_gpu_layers > 0:
            command.extend(['-ngl', str(args.n_gpu_layers)])
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        return result.stdout.strip()
        
    except FileNotFoundError as e:
        return str(e)
    except subprocess.CalledProcessError as e:
        error_message = f"LLM 호출 중 오류 발생:\n{e.stderr}"
        print(error_message)
        return ""

async def list_server_tools(server_url):
    """서버에서 사용 가능한 도구 목록을 가져옵니다."""
    try:
        async with Client(server_url) as client:
            tools = await client.list_tools()
            print(f"서버에서 사용 가능한 도구: {[tool.name for tool in tools]}")
            return tools
    except Exception as e:
        print(f"도구 목록을 가져오는데 실패했습니다: {e}")
        return []

async def query_mcp_server(command_input):
    """
    fastmcp 라이브러리를 사용하여 MCP 서버에 명령을 보내고 응답을 받습니다.
    """
    server_url = "http://localhost:8001/sse"

    try:
        parts = command_input.split()
        if not parts:
            return {"error": "명령어를 입력해주세요."}
            
        tool_name = parts[0][1:]  # '/' 제거
        tool_args = parts[1:]

        print(f"MCP 서버 {server_url}에 연결 중...")
        
        # async with 패턴 사용 (참고 코드와 동일한 방식)
        async with Client(server_url) as client:
            print("MCP 서버에 성공적으로 연결되었습니다.")
            
            # 먼저 사용 가능한 도구 목록 확인
            try:
                tools = await client.list_tools()
                available_tools = [tool.name for tool in tools]
                print(f"사용 가능한 도구들: {available_tools}")
                
                if tool_name not in available_tools:
                    return {"error": f"'{tool_name}' 도구를 찾을 수 없습니다. 사용 가능한 도구: {available_tools}"}
            except Exception as e:
                print(f"도구 목록 조회 실패: {e}")
            
            # 동적으로 도구 호출 (하드코딩 제거)
            if tool_name == "get_time":
                args_dict = {}
            elif tool_name == "get_weather":
                if len(tool_args) >= 1:
                    args_dict = {"location": tool_args[0]}
                else:
                    return {"error": "get_weather 도구에는 location 인자가 필요합니다."}
            elif tool_name == "get_user_info":
                if len(tool_args) >= 1:
                    args_dict = {"username": tool_args[0]}
                else:
                    return {"error": "get_user_info 도구에는 username 인자가 필요합니다."}
            else:
                # 알 수 없는 도구의 경우 기본적으로 시도
                args_dict = {}
                for i, arg in enumerate(tool_args):
                    args_dict[f"arg_{i}"] = arg
            
            print(f"도구 '{tool_name}' 호출 중, 인자: {args_dict}")
            result = await client.call_tool(tool_name, args_dict)
            
            # 참고 코드처럼 result.content[0].text로 접근 시도
            try:
                if hasattr(result, 'content') and len(result.content) > 0:
                    tool_output = result.content[0].text
                    return {"tool_output": tool_output, "raw_result": result.content}
                elif hasattr(result, 'data'):
                    return {"tool_output": str(result.data), "raw_result": result.data}
                else:
                    return {"tool_output": str(result), "raw_result": result}
            except (AttributeError, IndexError) as e:
                # 혹시 구조가 다를 경우 대비
                return {"error": f"응답 파싱 오류: {e}", "raw_result": str(result)}

    except ClientError as e:
        return {"error": f"MCP 서버 연결 실패: {server_url} - {e}"}
    except NotFoundError as e:
        return {"error": f"서버에 '{tool_name}' 도구가 존재하지 않습니다: {e}"}
    except ToolError as e:
        return {"error": f"도구 실행 중 오류 발생: {e}"}
    except Exception as e:
        return {"error": f"명령 처리 중 오류 발생: {e}"}

async def main_async():
    """
    단일 실행 MCP 클라이언트.
    하나의 사용자 입력을 처리하고 종료합니다.
    """
    parser = argparse.ArgumentParser(description='MCP Client with Local LLM support')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", default="./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", default=256)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", default=8)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", default=2048)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature for sampling", default=0.8)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="Number of layers to offload to GPU", default=0)
    parser.add_argument("-q", "--query", type=str, help="User query to process", default=None)
    
    args = parser.parse_args()

    print("==================================================")
    print("  MCP 클라이언트 (단일 실행 모드)")
    print("==================================================")
    
    # 명령행에서 질문을 받았는지 확인
    if args.query:
        user_input = args.query
        print(f"명령행 질문: {user_input}")
    else:
        user_input = input("질문을 입력하세요: ")
    
    # 기본 시스템 프롬프트 정의
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI agent assistant."

    try:
        if user_input.startswith('/'):
            # 특별 명령어 처리
            if user_input == "/tools":
                print("서버의 도구 목록을 조회 중...")
                tools = await list_server_tools("http://localhost:8001/sse")
                if tools:
                    print("사용 가능한 도구:")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("도구 목록을 가져올 수 없습니다.")
                return
            
            print("MCP 서버에 요청 중...")
            server_response = await query_mcp_server(user_input)
            print(f"서버 응답: {server_response}")

            # 서버 응답 요약
            if "tool_output" in server_response:
                tool_output = server_response["tool_output"]
                summary_prompt = f"""[User Question]
{user_input}

[Tool Execution Results]
{tool_output}

Please provide a concise and natural response in Korean to the user's question based on the tool execution results above.

[Final Answer]
"""
            else:
                summary_prompt = f"""사용자 질문: {user_input}
서버 응답에 오류가 있었습니다: {server_response.get('error', '알 수 없는 오류')}
이 상황을 사용자에게 친절하게 설명해주세요."""

            print("LLM이 서버 응답을 요약 중...")
            final_llm_response = query_local_llm(summary_prompt, args)
            print(f"\n최종 답변: {final_llm_response}")

        else:
            # 도구 호출 분석을 위한 LLM 프롬프트
            tool_call_prompt = f"""
            You are a tool calling assistant. Analyze the user's question and generate appropriate MCP tool call JSON.
            Available tools are as follows:
            - get_weather(location: str): Gets weather information for a specific location.
            - get_user_info(username: str): Gets information about a specific user.
            - get_time(): Gets the current time.

            User question: {user_input}
            
            If the question requires a tool, respond with JSON like: {{"tool": "get_weather", "args": ["서울"]}}
            If no tool is needed, respond with: {{"tool": null, "args": []}}
            
            Tool call JSON:
            """
            print("LLM이 도구 호출을 분석 중...")
            llm_tool_call_response = query_local_llm(tool_call_prompt, args)
            print(f"도구 분석 결과: {llm_tool_call_response}")
            
            try:
                parsed_tool_call = json.loads(llm_tool_call_response)
                tool_name = parsed_tool_call.get("tool")
                tool_args = parsed_tool_call.get("args", [])
                
                if tool_name and tool_name != "null":
                    mcp_command_input = f"/{tool_name} {' '.join(map(str, tool_args))}"
                    print(f"LLM이 '{mcp_command_input}' 명령을 제안했습니다")
                    
                    # 도구 실행
                    server_response = await query_mcp_server(mcp_command_input)
                    print(f"서버 응답: {server_response}")
                    
                    if "tool_output" in server_response:
                        tool_output = server_response["tool_output"]
                        summary_prompt = f"""[User Question]
{user_input}

[Tool Execution Results]
{tool_output}

Please provide a concise and natural response in Korean to the user's question based on the tool execution results above.

[Final Answer]
"""
                        print("LLM이 최종 답변을 생성 중...")
                        final_llm_response = query_local_llm(summary_prompt, args)
                        print(f"\n최종 답변: {final_llm_response}")
                    else:
                        print(f"도구 실행 오류: {server_response.get('error', '알 수 없는 오류')}")
                else:
                    # 일반 대화
                    prompt = f"{DEFAULT_SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
                    print("LLM이 응답을 생성 중...")
                    response = query_local_llm(prompt, args)
                    print(f"\n답변: {response}")

            except json.JSONDecodeError:
                print("LLM이 유효한 도구 호출 JSON을 생성하지 못했습니다. 일반 대화로 처리합니다.")
                prompt = f"{DEFAULT_SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
                response = query_local_llm(prompt, args)
                print(f"\n답변: {response}")

    except Exception as e:
        print(f"오류 발생: {e}")

    print("\n실행 완료.")

def main():
    """
    동기 main 함수에서 비동기 main_async 실행
    """
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()