"""Explain strict replay differences from float32 reduction memory order."""
from data import *

obs=pd.read_csv(OUT/'observation_table.csv')
frame=pd.read_csv(V1/'data/1_000300.csv')
indices=[0,500,1400,2500,3200,len(obs)-1]
cached=np.load(CACHE/'tokenizer_inputs.npy')
rows=[]
for i in indices:
    window=observed_window(frame,int(obs.anchor.iloc[i]))
    direct=tokenizer_transform(window)
    contiguous=tokenizer_transform(np.ascontiguousarray(window))
    rows.append(dict(index=i,pandas_array_strides=window.strides,
                     direct_max_error=float(np.max(np.abs(direct-cached[i]))),
                     contiguous_max_error=float(np.max(np.abs(contiguous-cached[i])))))
save_json(OUT/'normalization_layout_diagnostic.json',rows)
print(json.dumps(rows,indent=2))
