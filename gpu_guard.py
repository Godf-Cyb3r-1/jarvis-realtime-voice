"""GPU inventory and safe device mapping for Kaggle / CUDA / CTranslate2.

Logical CUDA indices are per process. They may differ from nvidia-smi indices
when CUDA_VISIBLE_DEVICES is set. Never assume GPU 1 exists.
"""
import os
import warnings

def inventory():
    import torch
    result={"torch_cuda_count":torch.cuda.device_count(),"devices":[]}
    try:
        import ctranslate2
        result["ctranslate2_cuda_count"]=ctranslate2.get_cuda_device_count()
    except Exception as error:
        result["ctranslate2_error"]=str(error)[:180]
    for i in range(result["torch_cuda_count"]):
        try:
            free,total=torch.cuda.mem_get_info(i)
            result["devices"].append({
                "index":i,"name":torch.cuda.get_device_name(i),
                "free_gb":round(free/2**30,2),"total_gb":round(total/2**30,2)
            })
        except Exception as error:
            result["devices"].append({"index":i,"error":str(error)[:150]})
    return result

def index_for(role, default=0, require_ctranslate=False):
    import torch
    count=torch.cuda.device_count()
    if require_ctranslate:
        try:
            import ctranslate2
            count=min(count,ctranslate2.get_cuda_device_count())
        except ImportError:
            raise RuntimeError("CTranslate2 is missing; install requirements-duplex-server.txt")
    if not count:
        raise RuntimeError(f"{role}: No compatible CUDA GPU visible. Check Kaggle accelerator and restart notebook.")
    value=os.environ.get(role)
    if value is None or value.strip().lower()=="auto":
        wanted=default if default<count else 0
    else:
        try:wanted=int(value)
        except ValueError:raise RuntimeError(f"{role} must be auto or a logical GPU index (0..{count-1})")
        if wanted<0 or wanted>=count:
            warnings.warn(f"{role}={wanted} is unavailable. Using GPU 0 of {count} visible GPU(s).")
            wanted=0
    return wanted

def print_inventory():
    import json
    print(json.dumps(inventory(),indent=2))
    print("CUDA_VISIBLE_DEVICES =",os.environ.get("CUDA_VISIBLE_DEVICES","(not set)"))
    if inventory().get("torch_cuda_count",0)<2:
        print("NOTE: Only one CUDA GPU visible; OmniVoice and Whisper will share GPU0.")
        print("Set LLM_BACKEND=api for best GPU memory headroom, or use a much smaller local model.")
if __name__=="__main__":
    print_inventory()
