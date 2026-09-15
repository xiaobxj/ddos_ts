from pathlib import Path
import json
P=Path(__file__).resolve().parents[1];O=P/'research_v32';N=P/'research_v34';N.mkdir(exist_ok=True)
assert not (N/'results/preparation_manifest.json').exists()
def write(n,s):(N/n).write_text(s,encoding='utf-8')
c0=json.loads((O/'protocol.json').read_text(encoding='utf-8'))
c={k:c0[k] for k in ['label_end','windows','seeds','neural','probe','offset_probe','report_python','decision_dates']}
policies=[dict(history='fixed_transform_step25',alpha=.25,role='primary'),dict(history='fixed_transform_step50',alpha=.5,role='sensitivity'),dict(history='fixed_transform_step100',alpha=1.,role='unrestricted_coefficient_control')]
pairs=[]
for p in policies:
    for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        for loss in ['direction_error','brier']:
            pairs.append(dict(history=p['history'],candidate=method,reference_history='rolling5_annual20',reference=method,metric=loss))
for ref in ['fixed_transform_step100','annual_net_quarter_head']:
    for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        for loss in ['direction_error','brier']:
            pairs.append(dict(history=policies[0]['history'],candidate=method,reference_history=ref,reference=method,metric=loss))
c.update(version=34,experiment='Annual transformations with fraction-limited quarterly coefficient updates',policies=c0['policies']+[p['history'] for p in policies],updates=policies,primary_comparisons_per_window=pairs,
  fixed_coordinates='Reuse R28 annual rolling5 natural20 networks, neural target scales, rep/market clipping and standardization, supervised annual R18 rays, vol/order residual projections, means and scales. No new neural fit, feature inference, scaler or interaction fit. Reuse verified R32 raw feature banks; apply the annual transformations to each quarter training and test row. Annual interactions need not remain orthogonal on the shifted quarter training sample; do not re-residualize.',
  fitting='At each nonDecember quarter end fit R18,R19,R23 ridge logistic coefficients separately on their frozen annual design matrices and same causal rolling5 daily members as R32. Original lambda=.01 penalizes slopes only, original float64 Newton tolerances. Fit R25 gamma on frozen annual order interaction with the newly fitted unrestricted same-quarter R19 as fixed offset; original gamma lambda=.01. The interaction rays remain the annual R18 ray even though the coefficient-only R18 is refit. Freeze all 204 fitted endpoints before weekly scoring.',
  update_rule='For each seed/method and cutoff, theta(alpha)=theta(annual)+alpha*(theta(unrestricted quarter endpoint)-theta(annual)), including intercept and gamma. alpha=.25 primary, .50 sensitivity, 1.00 fixed-transform unrestricted control; annual source is alpha0. Alpha fixed before result computation, not selected per year/method/seed. Anchor resets to new annual model each year; always shrink toward that annual anchor, not toward the previous quarterly update. Norm of drift is alpha times endpoint drift in these fixed standardized coordinates. This is a deterministic post-fit coefficient interpolation, not a separately optimized constrained logistic optimum. It implies interpolation of annual and quarter logits within a seed; never interpolate probabilities and never sigmoid(mean seed logit). Three seed probabilities retain equal mean.',
  offset_semantics='For each alpha the R25 nonorder coefficients and intercept equal the same-alpha R19 coefficients exactly. R25 gamma is interpolated between annual and unrestricted-quarter gamma. Intermediate gamma is not reoptimized conditional on interpolated R19. Record that distinction; only unrestricted endpoint is a scalar optimum.',
  causality=c0['head_update']['annual']+' '+c0['head_update']['training']+' '+c0['feature_bank'],
  scoring='All272 canonical weeks, all4 learned recipes plus native and training-frequency controls, all3 seeds. Q1 learned predictions copied exactly from annual baseline. Each candidate native is exact annual native, frequency exact full-quarterly frequency. Eight old policy predictions unchanged. 25% main candidate,50% sensitivity,100% control all reported with no retrospective winner selection.',
  inference='60 predefined contrasts jointly Holm: 2 periods * (3 alphas * 3 main methods * 2 losses versus annual + main25 versus fixed100 and R32 for 3 methods * 2 losses). Existing 10000 circular8-observation bootstrap seed20260910, centered two-sided p, marginal95CI. No across-round multiplicity adjustment; comparisons are exploratory on previously viewed history. Old R32 comparisons archived unchanged.',
  sequencing='Freeze protocol, all source code, inputs and4946oldfiles before fitting. Independently validate causal memberships, boundaries, source hashes, descriptors and synthetic coefficient/logit interpolation identities. Fit51 quarter-seed jobs /204 endpoints, construct612 policy coefficients and freeze before scoring. All3 policies evaluated once; no modification from observed weekly results. Independently solve endpoints, reconstruct frozen transforms, verify interpolation/R25 constraints, causal interfaces, probabilities, metrics and all60 statistical contrasts.',
  verification='Independent L-BFGS-B on153 logistic endpoints and Brent roots on51 gamma endpoints. Scalar coordinate reconstruction from immutable annual pipeline, training inputs from strict causal rows. Poison all out-of-training raw features, market descriptors, order descriptors and labels; selected fitting interface must not change. Verify annual sources mature, Q1 exact annual, alpha0 endpoints, alpha1 endpoints,612 coefficient drift ratios, three-seed mean probabilities, raw native no artificial probability, all old hashes and frozen artifact phase order.',
  budgets=dict(old_files=4946,annual_networks=18,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,new_head_jobs=51,new_endpoint_fits=204,derived_policy_heads=612,reused_annual_heads=72,weeks=272,model_rows=47872,ensemble_rows=17952),
  limits='All272 historical outcomes already viewed; motivated by R33 attribution and thus not independent blind test. No guarantee a smaller coefficient norm drift improves prediction or preserves annual accuracy. Some neural features in-sample for head training, inherited from baseline. This intervention jointly fixes transformations and shrinks coefficients; fixed100 control isolates shrinkage conditional on fixed coordinates. It does not prove the R33 market/order components cause errors, does not remove either component, and does not imply later out-of-sample market predictability. Preserve R25 old-repeat recent73/125=58.4%. No automatic promotion, execution, cost or return claims.')
write('protocol.json',json.dumps(c,indent=2,ensure_ascii=False))
s=(O/'evaluate32.py').read_text(encoding='utf-8').replace('common32','common34').replace('==28','==60').replace('primary_contrasts=28,histories=8','primary_contrasts=60,histories=11')
write('evaluate34.py',s)
s=(O/'contract32.py').read_text(encoding='utf-8').replace('common32','common34').replace('cases=boundary_cases();','cases=boundary_cases()+coefficient_cases();')
s=s.replace("def main():",'''def coefficient_cases():
    from scipy.special import expit
    a=np.array([-.7, .3, -.2]);b=np.array([.8, -.4, .6]);x=np.array([[1.,2.],[-1.,.5]])
    for alpha in [0.,.25,.5,1.]:
        t=shrink(a,b,alpha);np.testing.assert_allclose(design(x)@t,(1-alpha)*(design(x)@a)+alpha*(design(x)@b),rtol=0,atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(t-a),alpha*np.linalg.norm(b-a),rtol=0,atol=1e-15)
    np.testing.assert_array_equal(shrink(a,b,0),a);np.testing.assert_array_equal(shrink(a,b,1),b)
    assert abs(expit(.25*3+.75*(-1))-(.25*expit(3)+.75*expit(-1)))>.01
    z=np.array([-3.,1.,1.]);assert expit(z).mean()>.5 and expit(z.mean())<.5
    assert not probability(np.array([0.]))[0]>.5
    rng=np.random.default_rng(20260910);xx=[rng.normal(size=(90,n)) for n in [3,4,5]];xx.append(xx[-1]);y=(rng.random(90)>.48).astype(float)
    ts,_,ms=fit_coefficients(xx,y);np.testing.assert_array_equal(np.r_[ts[3][:-2],ts[3][-1]],ts[1])
    for j in range(3):assert ms[j]['gradient_inf']<=1e-9
    assert ms[3]['gradient_inf']<=1e-10
    return ['fractional_coefficient_drift','within_seed_logit_identity','exact_alpha0_alpha1','not_probability_blend','not_sigmoid_mean_logit','strict_half_is_down','synthetic_fixed_design_solvers_and_R25_offset']

def main():''')
write('contract34.py',s)
s=(O/'run32.py').read_text(encoding='utf-8').replace("'extract',",'').replace('{phase}32.py','{phase}34.py');write('run34.py',s)
s=(O/'delivery32.py').read_text(encoding='utf-8').replace('common32','common34').replace("'extraction',",'');write('delivery34.py',s)
write('README.md','# 第三十四轮：固定年度变换与受限季度系数更新\n\n主候选 25%，敏感性 50%，固定变换完整更新对照 100%。先冻结，再拟合与评分。所有历史已查看，不是新盲测。\n\n运行：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v34/run34.py prepare contract fit score evaluate verify report`。图像审查通过后单独运行 `delivery34.py`。\n')
print('R34 protocol and adapted modules written; no fitting or weekly scoring.')
