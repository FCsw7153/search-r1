"""Real local weights: generation, full-parameter backward and AdamW update."""
import argparse,json,time,os
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1];path=root/'data/models'/args.model
world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
torch.cuda.set_device(local_rank)
if world>1:torch.distributed.init_process_group('nccl')
torch.manual_seed(42);torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).cuda()
text=tokenizer.apply_chat_template([{'role':'user','content':'What is the capital of France?'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
batch=tokenizer(text,return_tensors='pt').to('cuda')
model.eval()
with torch.no_grad():out=model.generate(**batch,max_new_tokens=16,do_sample=False)
answer=tokenizer.decode(out[0,batch.input_ids.shape[1]:],skip_special_tokens=True);print('GENERATION',answer,flush=True)
model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
train_model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank]) if world>1 else model
result=train_model(**batch,labels=batch.input_ids,use_cache=False)
assert torch.isfinite(result.loss);result.loss.backward()
grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(grad_norm)
param=next(x for x in model.parameters() if x.grad is not None and x.grad.abs().max()>0)
before=param.detach().clone();optimizer.step();changed=bool(torch.any(param!=before));assert changed
report={'model':args.model,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'generation':answer,'loss':result.loss.item(),'grad_norm':grad_norm.item(),'optimizer_updated':changed,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'world_size':world,'scope':'BF16 full-parameter AdamW step; not Search-R1 RL'}
if rank==0:
 (root/'local_setup'/f'{args.model}-validation-{world}gpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if world>1:torch.distributed.destroy_process_group()
