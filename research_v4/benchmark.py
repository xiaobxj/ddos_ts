"""Synthetic throughput only, no market labels or model selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from architecture import *
import json
import time

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
batch=int(sys.argv[2]) if len(sys.argv)>2 else 32
results=[]
for dims in [5,20]:
    torch.manual_seed(20260910)
    model=SegmentedCrossformer(dims).to(device)
    x=torch.randn(batch,25,5,dims,device=device)
    geo=torch.rand(batch,25,2,device=device)
    valid=torch.ones(batch,25,dtype=torch.bool,device=device)
    valid[:,:12]=False
    y=torch.randn(batch,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    start=time.perf_counter()
    for i in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss=((model(x,geo,valid)-y)**2).mean()
        loss.backward();optimizer.step()
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    result=dict(device=device,dims=dims,batch=batch,steps=12,seconds=elapsed,seconds_per_step=elapsed/12,
                parameters=sum(p.numel() for p in model.parameters()))
    results.append(result);print(result,flush=True)
(ROOT/'sources').mkdir(exist_ok=True)
(ROOT/'sources'/f'synthetic_throughput_{device}_b{batch}.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
