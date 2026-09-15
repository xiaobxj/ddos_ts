"""Official frozen Kronos forecasts, price-free future calendar and seed audits."""
from common import *
import time
import importlib.metadata
import subprocess


def prepare_input(observed_frame, anchor, price_only):
    # Caller may pass only the observed prefix; no future rows needed.
    window = observed_frame.iloc[anchor-124:anchor+1]
    assert len(window) == 125
    values = window[['open', 'high', 'low', 'close', 'volume']].copy().reset_index(drop=True)
    if price_only:
        values['volume'] = 0.
    values['amount'] = 0.  # Missing channel, not measured zero turnover.
    timestamps = pd.to_datetime(window.date).reset_index(drop=True)
    return values, timestamps


def execution_forecast(path):
    opens = path.open.to_numpy(float)
    if not np.isfinite(opens).all() or min(opens[0], opens[-1]) <= 0:
        raise ValueError('Nonfinite or nonpositive predicted open; do not drop sample')
    return opens[-1] / opens[0] - 1


def invalid_bars(path):
    a = path[['open', 'high', 'low', 'close']]
    return int(((a.high < a[['open', 'close']].max(axis=1)) |
                (a.low > a[['open', 'close']].min(axis=1)) | (a.high < a.low)).sum())


def main():
    sys.path.insert(0, str(ROOT / 'vendor/Kronos'))
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = cfg(); settings = protocol['kronos']
    started = time.time()
    OUT.mkdir(exist_ok=True)
    (OUT / 'kronos_paths').mkdir(exist_ok=True)
    weights = json.loads((ROOT / 'sources/weights_manifest.json').read_text(encoding='utf-8'))
    for group in weights:
        for file in group['files']:
            assert sha(ROOT / file['file']) == file['sha256']
    code_revision = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/Kronos'), 'rev-parse', 'HEAD'], text=True).strip()
    assert code_revision == settings['code_revision']
    manifest = dict(started_utc=pd.Timestamp.now(tz='UTC').isoformat(), protocol_sha256=sha(ROOT/'protocol.json'),
                    source_sha256={name: sha(ROOT/name) for name in ['common.py', 'run_kronos.py']},
                    official_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'vendor/Kronos/model').glob('*.py')},
                    code_revision=code_revision, weights=weights, device='cpu', dtype='float32', threads=4,
                    executable=sys.executable, python=sys.version,
                    packages={name: importlib.metadata.version(name) for name in ['torch','numpy','pandas','einops','huggingface_hub','safetensors']},
                    settings=settings, previous_evidence=previous_evidence())
    save_json(OUT / 'kronos_run_manifest.json', manifest)
    tokenizer = KronosTokenizer.from_pretrained(str(ROOT/'models/Kronos-Tokenizer-base'), local_files_only=True)
    model = Kronos.from_pretrained(str(ROOT/'models/Kronos-small'), local_files_only=True)
    tokenizer.eval(); model.eval()
    assert not model.training and not tokenizer.training
    predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=settings['max_context'], clip=settings['clip'])
    frame = pd.read_csv(V1/'data/1_000300.csv')
    observation = observation_table(frame)
    old = pd.read_csv(V2/'results/predictions.csv')
    anchors = old[(old.method == 'f_multi_exp_quarter') & (old.date >= settings['common_start'])].anchor
    chosen = observation[observation.anchor.isin(anchors)].sort_values('anchor')
    # Separate calendar object passed to forecasts: dates only, no future prices.
    calendar = pd.to_datetime(frame.date)
    rows, seed_rows, audits = [], [], []
    repeatability_checked = False
    for row in chosen.itertuples(index=False):
        for name, price_only in [('kronos_ohlcv', False), ('kronos_ohlc_diagnostic', True)]:
            x, x_time = prepare_input(frame.iloc[:row.anchor+1], row.anchor, price_only)
            y_time = calendar.iloc[row.anchor+1:row.exit+1].reset_index(drop=True)
            horizon = len(y_time)
            assert horizon == row.exit - row.anchor and y_time.iloc[-1] == pd.Timestamp(row.exit_date)
            paths = []
            for base_seed in settings['seeds']:
                seed = base_seed + int(row.anchor)
                path_file = OUT/'kronos_paths'/f'{name}_{row.date}_{base_seed}.csv'
                # Resume only exact protocol/source checkpoints. Do not inspect scores here.
                torch.manual_seed(seed)
                with torch.inference_mode():
                    path = predictor.predict(x, x_time, y_time, pred_len=horizon,
                                             T=settings['temperature'], top_p=settings['top_p'], top_k=settings['top_k'],
                                             sample_count=settings['samples_per_seed'], verbose=False)
                assert np.isfinite(path.to_numpy()).all()
                if not repeatability_checked:
                    torch.manual_seed(seed)
                    with torch.inference_mode():
                        repeated = predictor.predict(x, x_time, y_time, pred_len=horizon,
                                                     T=settings['temperature'], top_p=settings['top_p'], top_k=settings['top_k'],
                                                     sample_count=settings['samples_per_seed'], verbose=False)
                    np.testing.assert_array_equal(path.to_numpy(), repeated.to_numpy())
                    repeatability_checked = True
                path.to_csv(path_file, index_label='forecast_date')
                paths.append(path.to_numpy())
                seed_rows.append(dict(base_seed=base_seed, effective_seed=seed, predicted_entry_open=float(path.open.iloc[0]),
                                      predicted_exit_open=float(path.open.iloc[-1]), invalid_bars=invalid_bars(path), horizon=horizon,
                                      **prediction_row(row, name, execution_forecast(path), settings['paper_pretrain_end'], 0)))
            average = pd.DataFrame(np.mean(paths, axis=0), columns=path.columns, index=y_time)
            average.to_csv(OUT/'kronos_paths'/f'{name}_{row.date}_ensemble.csv', index_label='forecast_date')
            rows.append(dict(predicted_entry_open=float(average.open.iloc[0]), predicted_exit_open=float(average.open.iloc[-1]),
                             invalid_bars=invalid_bars(average), horizon=horizon,
                             **prediction_row(row, name, execution_forecast(average), settings['paper_pretrain_end'], 0)))
            audits.append(dict(method=name, date=row.date, anchor=int(row.anchor), input_start=x_time.iloc[0].isoformat(),
                               input_end=x_time.iloc[-1].isoformat(), input_rows=len(x), missing_amount_zero=True,
                               price_only=price_only, future_calendar_start=y_time.iloc[0].isoformat(),
                               future_calendar_end=y_time.iloc[-1].isoformat(), horizon=horizon,
                               input_sha256=hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()))
        # Save checkpoints after each anchor, without aggregate scoring.
        pd.DataFrame(rows).to_csv(OUT/'kronos_predictions.csv', index=False)
        pd.DataFrame(seed_rows).to_csv(OUT/'kronos_seed_predictions.csv', index=False)
        save_json(OUT/'kronos_input_audit.json', audits)
        print(f'Kronos {row.date}: {len(rows)//2}/{len(chosen)} anchors; {time.time()-started:.1f}s', flush=True)
    assert previous_evidence() == manifest['previous_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time()-started,
                    anchors=len(chosen), arms=2, seeds_per_arm=3, deterministic_repeatability_passed=repeatability_checked,
                    previous_preserved=True)
    save_json(OUT/'kronos_run_manifest.json', manifest)
    print('Official Kronos inference completed.', flush=True)


if __name__ == '__main__':
    main()
