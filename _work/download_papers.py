from pathlib import Path
import concurrent.futures
import hashlib
import io
import json
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / '_work/pythonlibs'))
import requests
from pypdf import PdfReader

PAPERS = [
    dict(key='GPT4FTS', title='Beyond Fixed Patches: Enhancing GPTs for Financial Prediction with Adaptive Segmentation and Learnable Wavelets', authors='Renjun Jia; Zian Liu; Peng Zhu; Dawei Cheng; Yuqi Liang', version='arXiv:2505.02880v2 (2026-01-18); first submitted 2025-05-05', role='Adapted segmentation method used in article', landing='https://arxiv.org/abs/2505.02880v2', urls=['https://arxiv.org/pdf/2505.02880v2'], file='01_GPT4FTS_Beyond_Fixed_Patches_2505.02880v2.pdf', checks=['beyond fixed patches', 'learnable wavelets']),
    dict(key='Kronos', title='Kronos: A Foundation Model for the Language of Financial Markets', authors='Yu Shi; Zongliang Fu; Shuo Chen; Bohan Zhao; Wei Xu; Changshui Zhang; Jian Li', version='arXiv:2508.02739v1 (2025-08-02)', role='Tokenizer used; full model is comparison baseline', landing='https://arxiv.org/abs/2508.02739v1', urls=['https://arxiv.org/pdf/2508.02739v1'], file='02_Kronos_2508.02739v1.pdf', checks=['kronos', 'language of financial markets']),
    dict(key='Crossformer', title='Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting', authors='Yunhao Zhang; Junchi Yan', version='ICLR 2023', role='Backbone used in article', landing='https://openreview.net/forum?id=vSVLM2j9eie', urls=['https://openreview.net/pdf?id=vSVLM2j9eie'], file='03_Crossformer_ICLR2023.pdf', checks=['crossformer', 'cross-dimension']),
    dict(key='MERA', title='MERA: Mixture of Experts with Retrieval-Augmented Representation for Modeling Diversified Stock Patterns', authors='YuJun Liu; Chen-Hui Song; Peiyuan Liu; Naiqi Li; Tao Dai; Jigang Bao; Yong Jiang; Shu-Tao Xia', version='WWW Companion 2025; DOI:10.1145/3701716.3715513', role='Alternative encoder training reference, not reported used', landing='https://doi.org/10.1145/3701716.3715513', urls=['https://dl.acm.org/doi/pdf/10.1145/3701716.3715513'], file='04_MERA_WWW2025.pdf', checks=['mera', 'diversified stock patterns']),
    dict(key='TimeMosaic', title='TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding', authors='Kuiye Ding; Fanda Fan; Chunyi Hou; Zheya Wang; Lei Wang; Zhengxin Yang; Jianfeng Zhan', version='arXiv:2509.19406v5 (2026-01-05); first submitted 2025-09-23', role='Optional prediction-head reference, not reported used', landing='https://arxiv.org/abs/2509.19406v5', urls=['https://arxiv.org/pdf/2509.19406v5'], file='05_TimeMosaic_2509.19406v5.pdf', checks=['timemosaic', 'segment-wise decoding']),
]

def download(p):
    p = dict(p)
    p['attempts'] = []
    for url in p['urls']:
        try:
            r = requests.get(url, timeout=45, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            if not r.content.startswith(b'%PDF-'):
                raise ValueError('Response is not a PDF: ' + r.headers.get('Content-Type', ''))
            reader = PdfReader(io.BytesIO(r.content))
            pages = [page.extract_text() or '' for page in reader.pages]
            normalized = ' '.join(pages[0].lower().split())
            normalized = normalized.replace('\u00ad', '').replace('\u2010', '-').replace('\u2011', '-')
            compact = ''.join(c for c in normalized if c.isalnum())
            if not all(''.join(c for c in check.lower() if c.isalnum()) in compact for check in p['checks']):
                raise ValueError('First-page title did not match: ' + normalized[:250])
            dest = BASE / 'papers' / p['file']
            dest.write_bytes(r.content)
            (BASE / '_work' / (p['key'] + '.txt')).write_text('\n\n'.join(f'=== PAGE {i+1} ===\n{x}' for i,x in enumerate(pages)), encoding='utf-8')
            p.update(status='downloaded_and_title_verified', downloaded_url=r.url, pages=len(reader.pages), bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest(), retrieved_utc=datetime.now(timezone.utc).isoformat())
            p['attempts'].append({'url': url, 'status': r.status_code})
            break
        except Exception as e:
            p['attempts'].append({'url': url, 'error': str(e)})
            p['status'] = 'not_downloaded'
    print(json.dumps({'key':p['key'],'status':p['status'],'pages':p.get('pages'),'attempts':p['attempts']},ensure_ascii=False), flush=True)
    return p

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(download, PAPERS))
    (BASE / 'paper_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
