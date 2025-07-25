# BitNet-MCP
## Hybrid BitNet-MCP for Efficient Query Processing

### Build from source

> [!IMPORTANT]
> If you are using Windows, please remember to always use a Developer Command Prompt / PowerShell for VS2022 for the following commands. Please refer to the FAQs below if you see any issues.

1. Clone the repo
```bash
git clone --recursive https://github.com/YIDEUNKIM/BitNet-MCP.git
cd BitNet-MCP
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-mcp python=3.11.13
conda activate bitnet-mcp

cd BitNet
pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>
## Usage
### Basic usage

```bash
# Run MCP Server
cd ..
python mcp_server.py
```

```bash
# Run inference with the quantized model
cd BitNet
python mcp_client.py
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file (default="./models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text (default=256)
  -t THREADS, --threads THREADS
                        Number of threads to use(default=8)
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context (default=2048)
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature, a hyperparameter that controls the randomness of the generated text (default=0.8)
  -ngl, "--n-gpu-layers"
                        Number of layer to offload to GPU (dufault=0)
</pre>

### FAQ (Frequently Asked Questions)📌 

#### Q1: The build dies with errors building llama.cpp due to issues with std::chrono in log.cpp?

**A:**
This is an issue introduced in recent version of llama.cpp. Please refer to this [commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323) in the [discussion](https://github.com/abetlen/llama-cpp-python/issues/1942) to fix this issue.

#### Q2: How to build with clang in conda environment on windows?

**A:** 
Before building the project, verify your clang installation and access to Visual Studio tools by running:
```
clang -v
```

This command checks that you are using the correct version of clang and that the Visual Studio tools are available. If you see an error message such as:
```
'clang' is not recognized as an internal or external command, operable program or batch file.
```

It indicates that your command line window is not properly initialized for Visual Studio tools.

• If you are using Command Prompt, run:
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• If you are using Windows PowerShell, run the following commands:
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

These steps will initialize your environment and allow you to use the correct Visual Studio tools.


### git PR / commit

| **Type**         | **Description**                                   |
| ---------------- | ------------------------------------------------- |
| ✨**`Feat`**     | 새로운 기능 추가                                  |
| 🔨**`Fix`**      | 버그 수정                                         |
| 📝**`Docs`**     | 문서 작성 및 수정                                 |
| ⭐️**`Style`**   | 코드 스타일 및 포맷 변경(함수명/변수명 변경 포함) |
| 🧠**`Refactor`** | 코드 리팩토링(기능은 같으나 로직이 변경된 경우)   |
| **`Test`**       | 테스트 구현                                       |
| 🍎**`Chore`**    | 기타 수정 사항(ex: gitignore, application.yml)    |
| 🎨**`Design`**   | CSS 등 사용자 UI 디자인 변경                      |
| **`Comment`**    | 주석 작성 및 수정                                 |
| **`Rename`**     | 파일/폴더 명 수정 및 이동 작업                    |
| **`Remove`**     | 파일/폴더 삭제                                    |
| 🔥**`Hotfix`**   | 급하게 치명적인 버그를 고쳐야 하는 경우           |
