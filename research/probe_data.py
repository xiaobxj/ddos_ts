from concurrent.futures import ThreadPoolExecutor
import requests

URLS = [
    "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&klt=101&fqt=0&beg=20100101&end=20260831&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&lmt=10000",
    "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000300&klt=101&fqt=0&beg=20100101&end=20260831&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57,f58",
    "https://query1.finance.yahoo.com/v8/finance/chart/000300.SS?range=max&interval=1d",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,2010-01-01,2015-01-01,2000,qfq",
    "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData?symbol=sh000300&scale=240&ma=no&datalen=6000",
]


def probe(url):
    try:
        r = requests.get(url, timeout=12)
        return url, r.status_code, len(r.content), r.text[:220]
    except Exception as e:
        return url, type(e).__name__


for result in ThreadPoolExecutor(4).map(probe, URLS):
    print(result, flush=True)
