"""Independently re-encode representative historical windows and both weights."""
from util import *
from data import observed_window,tokenizer_transform

torch=initialize_torch()
sys.path.insert(0,str(V3/'vendor/Kronos'))
from model import KronosTokenizer

obs=pd.read_csv(OUT/'observation_table.csv')
frame=pd.read_csv(V1/'data/1_000300.csv')
indices=np.array([0,500,1400,2500,3200,len(obs)-1])
# prepare.py stacks windows into a C-contiguous array before normalization.
# Pandas gives column-contiguous windows; float32 reductions otherwise use a
# different summation order. Recreate the frozen layout, without loosening the
# exact comparison or changing any prepared input/model/prediction.
all_inputs=np.array([tokenizer_transform(np.ascontiguousarray(observed_window(frame,int(t)))) for t in obs.anchor])
np.testing.assert_array_equal(all_inputs,np.load(CACHE/'tokenizer_inputs.npy'))
inputs=all_inputs[indices]
models=[]
path=V3/'models/Kronos-Tokenizer-base'
models.append(('kronos',KronosTokenizer.from_pretrained(str(path),local_files_only=True).eval().cuda()))
random=KronosTokenizer(**json.loads((path/'config.json').read_text(encoding='utf-8')))
random.load_state_dict(torch.load(CACHE/'random_tokenizer_state.pt',map_location='cpu',weights_only=True))
models.append(('random_tokenizer',random.eval().cuda()))
cached=np.load(CACHE/'window_representations.npz')
for name,model in models:
    with torch.inference_mode():
        encoded=model.indices_to_bits(model.encode(torch.from_numpy(inputs).cuda(),half=True),half=True).cpu().numpy()
    np.testing.assert_array_equal(encoded,cached[name][indices])
save_json(OUT/'tokenizer_verification.json',dict(status='PASS',windows_per_model=len(indices),models=2,
                                               input_windows_independently_recomputed=len(obs),
                                               restored_preparation_C_contiguous_layout=True,bitwise_encoding_match=True))
print('Tokenizer cache independently re-encoded: PASS')
