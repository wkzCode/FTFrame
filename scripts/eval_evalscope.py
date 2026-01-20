import argparse
from evalscope import TaskConfig, run_task


def parse_args():
    p = argparse.ArgumentParser(description="Eval MMLU via EvalScope -> OpenAI API endpoint (vLLM server)")
    p.add_argument("--model", default="DeepSeek-Qwen3-8b", help="Model name shown in reports (string)")
    p.add_argument("--api-url", default="http://127.0.0.1:8801/v1/chat/completions", help="vLLM OpenAI-compatible endpoint")
    p.add_argument("--mmlu-path", required=True, help="Local path to MMLU dataset directory")

    p.add_argument("--batch", type=int, default=128, help="Eval batch size")
    p.add_argument("--timeout", type=int, default=60000, help="Timeout (ms)")
    p.add_argument("--stream", action="store_true", default=True, help="Use streaming output")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (for quick test)")
    p.add_argument("--use-cache", default=None, help="EvalScope cache directory")

    # Generation config (match your current defaults)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--enable-thinking", action="store_true", default=False, help="Enable Qwen thinking mode")

    return p.parse_args()


def main():
    args = parse_args()

    task_cfg = TaskConfig(
        model=args.model,
        api_url=args.api_url,
        eval_type='openai_api',
        datasets=['mmlu'],
        dataset_args={'mmlu': {'local_path': args.mmlu_path}},
        eval_batch_size=args.batch,
        generation_config={
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'n': args.n,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': args.enable_thinking}},
        },
        timeout=args.timeout,
        stream=args.stream,
        limit=args.limit,
        use_cache=args.use_cache,
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
