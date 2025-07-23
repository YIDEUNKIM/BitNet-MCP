#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // 데이터 손실 가능성
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

static void write_logfile(
    const llama_context * ctx, const common_params & params, const llama_model * model,
    const std::vector<llama_token> & input_tokens, const std::string & output,
    const std::vector<llama_token> & output_tokens
) {
    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = string_get_sortable_timestamp();

    const bool success = fs_create_directory_with_parents(params.logdir);
    if (!success) {
        LOG_ERR("%s: failed to create logdir %s, cannot write logfile\n", __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        LOG_ERR("%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: main\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    yaml_dump_non_result_info(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    yaml_dump_string_multiline(logfile, "output", output.c_str());
    yaml_dump_vector_int(logfile, "output_tokens", output_tokens);

    llama_perf_dump_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);

            // 모든 로그가 flush 되도록 보장
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

static std::string chat_add_and_format(struct llama_model * model, std::vector<common_chat_msg> & chat_msgs, const std::string & role, const std::string & content) {
    common_chat_msg new_msg{role, content};
    auto formatted = common_chat_format_single(model, g_params->chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({role, content});
    LOG_DBG("formatted: '%s'\n", formatted.c_str());
    return formatted;
}

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sparams;

    // 나중에 사용하기 위해 색상 사용 여부 저장
    // (추후 참고: 이 선택은 약간 어색함)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        LOG_ERR("************\n");
        LOG_ERR("%s: perplexity 계산에는 'perplexity' 도구를 사용해주세요\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: embedding 계산에는 'embedding' 도구를 사용해주세요\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: 경고: 최소 context 크기는 8이므로, 최소 크기를 사용합니다.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: 경고: RoPE frequency base를 %g로 변경합니다.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: 경고: RoPE frequency를 %g로 스케일링합니다.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama 백엔드 초기화\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    std::vector<common_chat_msg> chat_msgs;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    // 모델을 로드하고, lora 어댑터가 있다면 적용합니다
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model;
    ctx = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    LOG_INF("%s: llama threadpool 초기화, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // non-batch threadpool을 paused 상태로 시작
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: 모델이 %d개의 context 토큰으로만 훈련되었습니다 (%d 지정됨)\n", __func__, n_ctx_train, n_ctx);
    }

    // 대화 모드에서 chat template 예시 출력
    if (params.conversation) {
        if (params.enable_chat_template) {
            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(model, params.chat_template).c_str());
        } else {
            LOG_INF("%s: suffix/prefix가 지정되었으므로 chat template이 비활성화됩니다\n", __func__);
        }
    }

    // 시스템 정보 출력
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: '%s'에서 저장된 세션을 로드 시도 중\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: 세션 파일이 존재하지 않으므로 새로 생성합니다.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: 세션 파일이 비어있습니다. 새 세션을 초기화합니다.\n", __func__);
        } else {
            // 파일이 존재하며 비어있지 않습니다
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: '%s' 세션 파일 로드 실패\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: %d 토큰 크기의 프롬프트로 세션을 로드했습니다\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    {
        auto prompt = (params.conversation && params.enable_chat_template && !params.prompt.empty())
            ? chat_add_and_format(model, chat_msgs, "system", params.prompt) // 대화 모드에서 system 프롬프트 포맷팅
            : params.prompt;
        if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
            LOG_DBG("프롬프트 토큰화\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("세션 토큰 사용\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // 토큰 없이 실행해서는 안 됩니다
    if (embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_token_bos(model));
            LOG_WRN("embd_inp가 비어있는 것으로 간주되어 bos가 추가되었습니다: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("입력이 비어있습니다\n");
            return -1;
        }
    }

    // 부정 프롬프트 토큰화
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: 프롬프트가 너무 깁니다 (%d 토큰, 최대 %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // 해당되는 경우, 저장된 세션의 유사도에 대한 디버그 메시지
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: 세션 파일의 전체 프롬프트를 사용합니다\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: 세션 파일이 프롬프트와 정확히 일치합니다!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: 세션 파일이 프롬프트와 유사도가 낮습니다 (%zu / %zu 토큰); 대부분 재평가됩니다\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: 세션 파일이 프롬프트의 %zu / %zu 토큰과 일치합니다\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // 이전 세션에서 상속했을 수 있는 "미래" 토큰들을 제거합니다
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    LOG_DBG("캐시된 logits 재계산 (확인): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // 캐시의 끝에 도달하지 않고 전체 프롬프트에 캐시를 사용할 경우,
    // 마지막 토큰을 강제로 재평가하여 캐시된 logits를 재계산합니다
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // context를 리셋할 때 유지할 토큰 수
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // 항상 BOS 토큰을 유지합니다
    }

    if (params.conversation) {
        params.interactive_first = true;
    }

    // interactive 시작이 지정된 경우 interactive 모드를 활성화합니다
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: n_keep 기반의 정적 프롬프트: ''", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C 처리
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_INF("%s: interactive 모드 켜짐.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("BOS와 함께 입력 접두사\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("입력 접두사: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("입력 접미사: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: sampling 서브시스템 초기화 실패\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention 상태
    // 지금까지 그룹화된 KV 토큰 수 (params.grp_attn_n > 1일 경우에만 사용)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== interactive 모드로 실행 중입니다. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - 언제든지 Ctrl+C를 눌러 중단할 수 있습니다.\n");
#endif
        LOG_INF(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // 현재 assistant 메시지를 저장하기 위함, 대화 모드에서 사용

    // 가장 먼저 프롬프트를 출력할 것이므로, 그에 맞게 색상을 설정합니다
    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // 토큰화된 antiprompt
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params.antiprompt.size());
    for (const std::string & antiprompt : params.antiprompt) {
        antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size, 0, 0))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // 예측
        if (!embd.empty()) {
            // 참고: 여기서 (n_ctx - 4)는 동일한 값을 사용하는 --prompt 또는 --file을 통한
            // 커맨드라인 프롬프트 처리 로직과 맞추기 위함입니다.
            int max_embd_size = n_ctx - 4;

            // 필요한 경우 embd를 잘라내어 입력이 context 크기를 초과하지 않도록 합니다.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                // context shifting을 통한 무한 텍스트 생성
                // context가 부족할 경우:
                // - 원본 프롬프트에서 n_keep 만큼의 첫 토큰들을 가져옵니다 (n_past를 통해)
                // - 마지막 (n_ctx - n_keep) 토큰의 절반을 가져와 배치 단위로 logits를 재계산합니다

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_DBG("\n\n%s: context가 가득 찼고 context shift가 비활성화되어 중지합니다\n", __func__);
                        break;
                    } else {
                        if (params.n_predict == -2) {
                            LOG_DBG("\n\n%s: context가 가득 찼고 n_predict == -%d 이므로 중지합니다\n", __func__, params.n_predict);
                            break;
                        }

                        const int n_left    = n_past - params.n_keep;
                        const int n_discard = n_left/2;

                        LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                                n_past, n_left, n_ctx, params.n_keep, n_discard);

                        llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                        llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                        n_past -= n_discard;

                        LOG_DBG("after swap: n_past = %d\n", n_past);

                        LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                        LOG_DBG("clear session path\n");
                        path_session.clear();
                    }
                }
            } else {
                // Self-Extend를 통한 context 확장
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // 재평가 대신 로드된 세션에서 일치하는 접두사를 재사용하려고 시도합니다 (n_past를 통해)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) { // 
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                // 총 시간과 함께 총 토큰 수를 표시합니다
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // 선택적으로 첫 샘플에서 세션을 저장합니다 (다음 번 프롬프트 로딩 속도 향상을 위해)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // 이것을 콘솔에 echo
            input_echo = true;

            // 남은 sampling 예산 감소
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // 프롬프트나 상호작용에서 남은 사용자 입력이 있으므로, 처리 과정으로 전달합니다
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // 나중에 반복 패널티를 적용하기 위해 프롬프트를 sampling context에 푸시합니다
                // 프롬프트에는 문법 규칙을 적용하지 않습니다
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // 텍스트 표시
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // 콘솔/스트림 출력
                LOG("%s", token_str.c_str());

                // 표시된 토큰을 로그에 기록
                // 참고: 생성된 토큰은 하나씩 만들어지므로 이 확인이 필요합니다
                if (embd.size() > 1) {
                    // 들어오는 요청된 토큰
                    input_tokens.push_back(id);
                } else {
                    // 나가는 생성된 토큰
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // 대기 중인 사용자 입력이 없으면 색상을 기본값으로 재설정합니다
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }

        // 현재 대기열에 있는 입력을 처리하고 있지 않다면;
        if ((int) embd_inp.size() <= n_consumed) {
            // 마지막 n_prev 토큰에서 reverse prompt를 확인합니다
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // 각 reverse prompt가 출력의 끝에 나타나는지 확인합니다.
                // interactive 모드가 아닐 경우, reverse prompt는 뒤따르는 일부 문자와 함께 토큰화될 수 있으므로
                // 검색 범위를 약간 넓혀서 이를 보정합니다.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // 특수 토큰을 사용하여 reverse prompt를 확인합니다
                llama_token last_token = common_sampler_last(smpl);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // interactive 모드에서 생성 종료(EOG) 토큰을 처리합니다
            if (llama_token_is_eog(model, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // 첫 번째 reverse prompt를 토큰화하고 주입합니다
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format(model, chat_msgs, "assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            // 현재 토큰이 EOG가 아니면, 현재 assistant 메시지에 추가합니다
            if (params.conversation) {
                const auto id = common_sampler_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            if (n_past > 0 && is_interacting) {
                LOG_DBG("사용자 입력을 기다리는 중\n");

                if (params.conversation) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("입력 접두사 BOS 토큰 추가 중\n");
                    embd_inp.push_back(llama_token_bos(model));
                }
                
                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation) {
                    LOG_DBG("입력 접두사 추가 중: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // 사용자 입력에만 색상 적용
                console::set_display(console::user_input); // 색상 적용 별로 안중요함
                display = params.display_prompt;
                
                // 이 라인 mcp_client.py 에서 입력 파싱
                // ********************************* //
                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    // parma.multiline_input: 이 파라미터는 "사용자가 Enter 키를 눌렀을 때, 입력을 즉시 제출할 것인가, 아니면 그냥 줄바꿈만 하고 계속 입력을 받을 것인가?"를 제어합니다.
                    buffer += line;
                } while (another_line);
                // ********************************* //
                
                // 입력 완료, 색상 초기화
                console::set_display(console::reset);
                display = true;

                // ===================================================================================
                // ★★★ MCP 클라이언트 제어 지점 ★★★
                // 이 곳이 MCP 서버로 요청을 보낼지, 아니면 로컬 LLM으로 처리할지를 결정하는 최적의 위치입니다.
                // 'buffer' 변수에는 사용자가 입력한 순수 텍스트가 그대로 담겨 있습니다.
                //
                // if (buffer.rfind("/", 0) == 0) {
                //     // 1. MCP 서버용 명령이라고 판단될 경우 (예: '/'로 시작)
                //     // 2. 여기서 buffer의 내용을 파싱하여 서버에 보낼 JSON 요청을 만듭니다.
                //     // 3. 네트워킹 라이브러리(libcurl 등)를 사용해 서버와 통신합니다.
                //     // 4. 서버의 응답을 받아 사용자에게 출력합니다.
                //     // 5. 아래의 else 블록(로컬 LLM 처리)을 건너뛰고 다음 입력을 기다립니다.
                // } else {
                //     // MCP 명령이 아닐 경우, 기존 로직을 그대로 실행하여 로컬 LLM이 응답을 생성합니다.
                //     // (아래의 모든 코드를 이 else 블록 안으로 옮겨야 합니다.)
                // }
                // ===================================================================================
                
                // 입력 버퍼가 비어있지 않은 경우에만 embd에 토큰을 추가합니다
                // 빈 줄을 입력하면 사용자가 제어권을 다시 넘겨줄 수 있습니다
                if (buffer.length() > 1) {
                    // 입력 접미사가 있으면 추가합니다
                    if (!params.input_suffix.empty() && !params.conversation) {
                        LOG_DBG("입력 접미사 추가 중: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }
                
                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }
                    
                    bool format_chat = params.conversation && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format(model, chat_msgs, "user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: 현재 chat template 구현의 한 가지 불편한 점은 사용자 입력과 특수 토큰(접두사/접미사)을 구분할 수 없다는 것입니다
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());
                    
                    // 사용자가 중간에 생성을 멈추면, 모델의 마지막 응답을 끝내기 위해 EOT를 추가해야 합니다
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_token_eot(model);
                        embd_inp.push_back(eot == -1 ? llama_token_eos(model) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    // assistant 메시지 초기화
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                } else {
                    LOG_DBG("빈 줄, 제어권을 다시 넘깁니다\n");
                }

                input_echo = false; // 이것을 다시 echo하지 마세요
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }

        // 생성 종료
        if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // interactive 모드에서는 최대 토큰 수를 존중하고, 도달하면 사용자 입력으로 돌아갑니다.
        // n_predict가 -1(무한) 또는 -2(context 크기에서 중지)일 때는 이 로직을 건너뜁니다.
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: 최종 출력을 세션 파일 '%s'에 저장 중\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);
    write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    common_sampler_free(smpl);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    ggml_threadpool_free(threadpool);
    ggml_threadpool_free(threadpool_batch);

    return 0;
}

