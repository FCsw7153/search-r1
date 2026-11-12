"""One local vLLM generation using the unified environment; no training job."""
def main():
    import importlib.metadata as metadata
    from pathlib import Path
    for name in ['torch','vllm','transformers','tensordict','TransferQueue','torchdata','flash-attn','faiss-cpu']:
        print('VERSION',name,metadata.version(name),flush=True)
    import torch,faiss,transfer_queue,torchdata
    from verl.trainer.main_ppo import main as trainer_main
    print('VERL TRAINER IMPORT OK',flush=True)
    from transformers import AutoConfig,AutoTokenizer
    root=Path(__file__).resolve().parents[1]
    for name in ['Qwen3.5-2B','Qwen3.5-4B']:
        cfg=AutoConfig.from_pretrained(root/'data/models'/name,local_files_only=True)
        print('LOCAL MODEL',name,cfg.model_type,flush=True)
    from vllm import LLM,SamplingParams
    model=str(root/'data/models/Qwen3.5-2B')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'What is the capital of France? Answer briefly.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    llm=LLM(model=model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=1024,max_num_batched_tokens=1024,max_num_seqs=1,gpu_memory_utilization=0.35,enforce_eager=True,enable_prefix_caching=False,limit_mm_per_prompt={'image':0,'video':0})
    results=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=16))
    print('VLLM LOCAL GENERATION',results[0].outputs[0].text,flush=True)
    print('UNIFIED ENV CHECK PASSED',flush=True)
if __name__=='__main__':
    main()
