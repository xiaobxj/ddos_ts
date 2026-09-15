from common48 import *
def main():
    check_frozen();assert not (OUT/'assembly_manifest.json').exists();run=manifest('assembly');heads=assemble_records(read(OUT/'uniform_heads.json'),read(OUT/'weighted_heads.json'));assert len(heads)==144;files=[]
    p=OUT/'heads.json';save(p,heads);files.append(p);rows=[]
    for h in heads:
        for j,value in enumerate(h['coefficients']):rows.append(dict(history=h['history'],cutoff=h['cutoff'],seed=h['seed'],method=h['method'],coordinate=j,role='intercept' if j==len(h['coefficients'])-1 else 'slope',source=h['intercept_source'] if j==len(h['coefficients'])-1 else h['slope_source'],value=value))
    p=OUT/'coefficient_sources.csv';pd.DataFrame(rows).to_csv(p,index=False);files.append(p);check_frozen();finish(run,files,assembled_heads=144,new_head_fits=0,new_neural_fits=0,labels_used=False,weekly_scoring_during_assembly=False);print('144fixed hybrid heads frozen;no fitting or weekly scoring.',flush=True)
if __name__=='__main__':main()
