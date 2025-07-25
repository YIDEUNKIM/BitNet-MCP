import argparse
import subprocess
import os
import platform
import json
import asyncio

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from fastmcp import Client
from fastmcp.exceptions import ClientError, ToolError, NotFoundError


# 3. 설정 관리 개선
@dataclass
class AppConfig:
    """애플리케이션 설정을 관리하는 데이터 클래스"""
    server_url: str = "http://localhost:8001/sse"
    default_model_path: str = "./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    default_n_predict: int = 256
    default_threads: int = 8
    default_ctx_size: int = 2048
    default_temperature: float = 0.8
    default_n_gpu_layers: int = 0
    default_system_prompt: str = "You are a intelligent AI agent assistant."

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
    # 디버깅: LLM에 전달되는 프롬프트 출력
    print("\n[DEBUG] --- LLM Prompt ---")
    print(prompt)
    print("---------------------------")

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
        error_message = f"LLM 호출 중 오류 발생:\n"
        print(error_message)
        return ""

async def list_server_tools(server_url: str) -> Optional[List[Any]]:
    """서버에서 사용 가능한 도구 목록을 가져옵니다."""
    try:
        async with Client(server_url) as client:
            tools = await client.list_tools()
            return tools
    except Exception as e:
        print(f"도구 목록을 가져오는데 실패했습니다: {e}")
        return None

# 1. query_mcp_server 함수 리팩토링
import re

async def query_mcp_server(command_input: str, server_url: str, available_tools: List[Any]) -> Dict[str, Any]:
    """
    fastmcp 라이브러리를 사용하여 MCP 서버에 명령을 보내고 응답을 받습니다.
    도구 인자를 동적으로 처리하고, 입력 문자열을 정리합니다.
    """
    try:
        # 입력 문자열의 양 끝 공백 제거
        command_input = command_input.strip()
        if not command_input:
            return {"error": "명령어를 입력해주세요."}

        parts = command_input.split()
        command_part = parts[0]
        tool_args = parts[1:]

        if not command_part.startswith('/'):
            return {"error": "명령어는 /로 시작해야 합니다."}

        # 정규표현식을 사용하여 명령어에서 알파벳, 숫자, 밑줄(_)만 남기고 모두 제거
        tool_name = re.sub(r'[^a-zA-Z0-9_]', '', command_part[1:])

        # 사용 가능한 도구 목록에서 요청된 도구 찾기
        tool_to_call = next((tool for tool in available_tools if tool.name == tool_name), None)

        if not tool_to_call:
            available_tool_names = [tool.name for tool in available_tools]
            # 디버깅을 위해 원본 명령어와 정리된 명령어를 모두 보여줌
            return {"error": f"'{command_part[1:]}' (정리 후: '{tool_name}') 도구를 찾을 수 없습니다. 사용 가능한 도구: {available_tool_names}"}

        # 도구의 파라미터 정보 가져오기
        try:
            param_properties = tool_to_call.parameters.get('properties', {})
            param_names = list(param_properties.keys())
        except AttributeError:
            param_names = []

        # 인자 개수 정확히 확인
        if len(tool_args) != len(param_names):
             return {"error": f"{tool_name} 도구는 {len(param_names)}개의 인자가 필요하지만, {len(tool_args)}개가 제공되었습니다. (필요: {param_names})"}

        args_dict = dict(zip(param_names, tool_args))

        print(f"MCP 서버 {server_url}에 연결 중...")
        async with Client(server_url) as client:
            print("MCP 서버에 성공적으로 연결되었습니다.")
            print(f"도구 '{tool_name}' 호출 중, 인자: {args_dict}")
            result = await client.call_tool(tool_name, args_dict)
            
            try:
                if hasattr(result, 'content') and len(result.content) > 0:
                    tool_output = result.content[0].text
                    return {"tool_output": tool_output, "raw_result": result.content}
                elif hasattr(result, 'data'):
                    return {"tool_output": str(result.data), "raw_result": result.data}
                else:
                    return {"tool_output": str(result), "raw_result": result}
            except (AttributeError, IndexError) as e:
                return {"error": f"응답 파싱 오류: {e}", "raw_result": str(result)}

    except ClientError as e:
        return {"error": f"MCP 서버 연결 실패: {server_url} - {e}"}
    except NotFoundError as e:
        return {"error": f"서버에 '{tool_name}' 도구가 존재하지 않습니다: {e}"}
    except ToolError as e:
        return {"error": f"도구 실행 중 오류 발생: {e}"}
    except Exception as e:
        return {"error": f"명령 처리 중 오류 발생: {e}"}

def generate_tool_call_prompt(user_input: str, tools: List[Any]) -> str:
    """
    사용 가능한 도구 목록을 기반으로 LLM의 도구 호출 프롬프트를 동적으로 생성합니다.
    """
    if not tools:
        return ""
        
    tool_descriptions = []
    for tool in tools:
        # 파라미터 정보를 좀 더 상세히 표시
        try:
            param_properties = tool.parameters.get('properties', {})
            params = ", ".join([f"{name}: {p.get('type', 'any')}" for name, p in param_properties.items()])
        except AttributeError:
            params = ""
        tool_descriptions.append(f"- {tool.name}({params}): {tool.description}")

    tools_formatted = "\n".join(tool_descriptions)

    return f"""
You are a tool calling assistant. Analyze the user's question and generate appropriate MCP tool call JSON.
Available tools are as follows:
{tools_formatted}

User question: {user_input}

If the question requires a tool, respond with JSON like: {{\"tool\": \"get_weather\", \"args\": [\"서울\"]}}
If no tool is needed, respond with: {{\"tool\": null, \"args\": []}}

Tool call JSON:
"""

async def main_async():
    """
    단일 실행 MCP 클라이언트.
    하나의 사용자 입력을 처리하고 종료합니다.
    """
    config = AppConfig()
    
    parser = argparse.ArgumentParser(description='MCP Client with Local LLM support')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", default=config.default_model_path)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", default=config.default_n_predict)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", default=config.default_threads)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", default=config.default_ctx_size)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature for sampling", default=config.default_temperature)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="Number of layers to offload to GPU", default=config.default_n_gpu_layers)
    parser.add_argument("-q", "--query", type=str, help="User query to process", default=None)
    
    args = parser.parse_args()

    print("====================================================")
    print("  MCP 클라이언트 (단일 실행 모드) - 리팩토링 버전")
    print("====================================================")

    # 서버에서 사용 가능한 도구 목록 가져오기
    print("서버에서 도구 목록을 가져오는 중...")
    available_tools = await list_server_tools(config.server_url)
    if available_tools:
        tool_names = [tool.name for tool in available_tools]
        print(f"사용 가능한 도구: {tool_names}")
    else:
        print("경고: 서버에서 도구 목록을 가져올 수 없습니다. 도구 관련 기능이 제한될 수 있습니다.")
        available_tools = []

    # 명령행에서 질문을 받았는지 확인
    if args.query:
        user_input = args.query
        print(f"명령행 질문: {user_input}")
    else:
        user_input = input("질문을 입력하세요: ")
    
    try:
        if user_input.startswith('/'):
            # 특별 명령어 처리
            if user_input == "/tools":
                if available_tools:
                    print("사용 가능한 도구:")
                    for tool in available_tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("도구 목록을 가져올 수 없습니다.")
                return
            
            print("MCP 서버에 요청 중...")
            server_response = await query_mcp_server(user_input, config.server_url, available_tools)
            print(f"서버 응답: {server_response}")

            # 서버 응답에 오류가 있는지 먼저 확인
            if "error" in server_response:
                print(f"\n오류: {server_response['error']}")
            elif "tool_output" in server_response:
                # 성공적인 도구 실행 결과만 LLM으로 요약
                tool_output = server_response["tool_output"]
                summary_prompt = f"""[User Question]
{user_input}

[Tool Execution Results]
{tool_output}

Please provide a concise and natural response in Korean to the user's question based on the tool execution results above.

[Final Answer]
"""
                print("LLM이 서버 응답을 요약 중...")
                final_llm_response_full = query_local_llm(summary_prompt, args)
                # LLM 응답에서 첫 번째 줄만 최종 답변으로 사용
                final_llm_response = final_llm_response_full.split('\n')[0].strip()
                print(f"\n최종 답변: {final_llm_response}")


        else:
            # '/'로 시작하지 않는 모든 입력은 일반 대화로 처리
            print("일반 대화로 처리합니다...")
            # 도구 호출과 동일한 구조화된 프롬프트 사용
            structured_prompt = f"""
        [System]
        {config.default_system_prompt}
        [User Question]
        {user_input}
        Please provide a concise and natural response to the user's question.
        [Final Answer]
        
        """

            response = query_local_llm(structured_prompt, args)

            # [Final Answer] 이후의 첫 번째 줄만 추출 (도구 호출과 동일한 방식)

            final_response = response.split('\n')[0].strip()

            print(f"\n답변: {final_response}")

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
