from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='DeepSeek-Qwen3-8b',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    eval_type='openai_api',
    datasets=[
        'mmlu',
    ],
    dataset_args={
        'mmlu': {
            'local_path': '',
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 20000,  # Max number of generated tokens, suggested to set a large value to avoid output truncation
        'temperature': 0.7,  # Sampling temperature (recommended value per Qwen report)
        'top_p': 0.8,  # top-p sampling (recommended value per Qwen report)
        'top_k': 20,  # top-k sampling (recommended value per Qwen report)
        'n': 1,  # Number of replies generated per request
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # close thinking mode
    },
    timeout=60000,  # Timeout
    stream=True,  # Use streaming output
    limit=None,  # Set to 1000 samples for testing
    # use_cache=""
)

run_task(task_cfg=task_cfg)