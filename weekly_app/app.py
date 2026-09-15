"""Loopback-only UI server. Opening the app never starts a research cycle."""
import argparse
import ctypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STATE = PROJECT / 'prospective_app'
PYTHON = PROJECT / 'research_v4/.venv_gpu/Scripts/python.exe'
APP_ID = 'ddos-weekly-local-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.stopping = False
        self.state = dict(busy=False, job=None, data=None, result=None, error=None, log=[], revision=0)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def add_log(self, message, stage=None, state=None):
        with self.lock:
            self.state['log'].append(dict(at=now(), message=message[:1500], stage=stage, state=state))
            self.state['log'] = self.state['log'][-180:]

    def start(self, action):
        if action not in ['inspect', 'cycle']:
            raise ValueError('Unknown action')
        with self.lock:
            if self.state['busy'] or self.stopping:
                return False
            self.state.update(busy=True, job=action, result=None, error=None, log=[], started_utc=now())
        threading.Thread(target=self.run, args=(action,), daemon=False).start()
        return True

    def request_stop(self):
        with self.lock:
            if self.state['busy']:
                return False
            self.stopping = True
            return True

    def run(self, action):
        got_result = False
        filename = STATE / 'logs' / (datetime.now().strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(4) + '.log')
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
            with filename.open('x', encoding='utf-8') as log:
                process = subprocess.Popen([str(PYTHON), '-B', '-u', str(ROOT / 'worker.py'), action],
                    cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding='utf-8', errors='replace', env=env,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                for line in process.stdout:
                    log.write(line); log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        if line.strip():
                            self.add_log(line.strip())
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get('event') == 'progress':
                        self.add_log(event['message'], event.get('stage'), event.get('state'))
                    elif event.get('event') == 'result':
                        got_result = True
                        with self.lock:
                            if event.get('data') is not None:
                                self.state['data'] = event['data']
                            if action == 'cycle':
                                self.state['result'] = event.get('result')
                            if not event['ok']:
                                self.state['error'] = event.get('error') or (event.get('result') or {}).get('error') or '流程未完成'
                code = process.wait()
                if not got_result:
                    raise RuntimeError(f'运行进程提前退出（代码 {code}），请查看运行日志。')
        except Exception as exc:
            with self.lock:
                self.state['error'] = str(exc)
            self.add_log(str(exc), state='failed')
        finally:
            with self.lock:
                self.state.update(busy=False, finished_utc=now(), log_file=str(filename))
                self.state['revision'] += 1


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, jobs, token):
        super().__init__(('127.0.0.1', 0), Handler)
        self.jobs, self.token = jobs, token
        self.last_seen = time.monotonic()
        self.origin = f'http://127.0.0.1:{self.server_port}'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, code, value, mime='application/json; charset=utf-8'):
        raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self):
        return (self.headers.get('Host') == self.server.origin.removeprefix('http://')
                and self.headers.get('X-App-Token') == self.server.token
                and self.headers.get('Origin', self.server.origin) == self.server.origin)

    def do_GET(self):
        if self.path.startswith('/api/'):
            if not self.allowed():
                return self.respond(403, dict(error='请通过本机启动入口打开程序。'))
            self.server.last_seen = time.monotonic()
            if self.path == '/api/state':
                return self.respond(200, dict(app_id=APP_ID, **self.server.jobs.snapshot()))
            return self.respond(404, {})
        names = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path not in names or self.headers.get('Host') != self.server.origin.removeprefix('http://'):
            return self.respond(404, {})
        name, mime = names[self.path]
        return self.respond(200, (ROOT / 'static' / name).read_bytes(), mime)

    def do_POST(self):
        if not self.allowed():
            return self.respond(403, dict(error='请求来源无效。'))
        if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length', '0') != '0':
            return self.respond(400, dict(error='此入口不接受日期、参数或账本覆盖。'))
        self.server.last_seen = time.monotonic()
        if self.path in ['/api/inspect', '/api/cycle']:
            action = self.path.rsplit('/', 1)[-1]
            started = self.server.jobs.start(action)
            return self.respond(202 if started else 409, dict(started=started, error=None if started else '已有任务正在运行，请等待完成。'))
        if self.path == '/api/quit':
            if not self.server.jobs.request_stop():
                return self.respond(409, dict(error='任务仍在运行，请完成后退出。'))
            self.respond(200, dict(stopped=True))
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        return self.respond(404, {})


def process_alive(pid):
    if os.name != 'nt':
        return Path(f'/proc/{pid}').exists()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        # Access denied is not evidence that a process has exited.
        return ctypes.get_last_error() != 87
    code = ctypes.c_ulong()
    try:
        return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)


def reuse(open_browser):
    descriptor = STATE / 'server.json'
    for _ in range(20):
        try:
            info = json.loads(descriptor.read_text(encoding='utf-8'))
            if info['app_id'] != APP_ID or not info['origin'].startswith('http://127.0.0.1:'):
                return False
            req = urllib.request.Request(info['origin'] + '/api/state', headers={'X-App-Token': info['token']})
            with urllib.request.urlopen(req, timeout=1) as response:
                if json.load(response)['app_id'] == APP_ID:
                    if open_browser:
                        webbrowser.open(info['origin'] + '/#' + info['token'])
                    return True
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            time.sleep(.15)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true', help='Start the local UI without opening a browser tab')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock = STATE / 'app.lock'
    try:
        with lock.open('x', encoding='ascii') as file:
            file.write(str(os.getpid()))
    except FileExistsError:
        if reuse(not args.no_browser):
            return
        pid = int(lock.read_text(encoding='ascii'))
        if process_alive(pid):
            raise RuntimeError('程序已启动但暂时无法连接，请稍后重新打开。')
        lock.unlink()  # Only this application's confirmed exited-process lock.
        return main()
    server = None
    try:
        jobs = Jobs()
        server = Server(jobs, secrets.token_urlsafe(32))
        write_json(STATE / 'server.json', dict(app_id=APP_ID, origin=server.origin, token=server.token, pid=os.getpid(), started_utc=now()))
        jobs.start('inspect')
        def retire_when_closed():
            while True:
                time.sleep(10)
                if time.monotonic() - server.last_seen > 120 and jobs.request_stop():
                    server.shutdown()
                    return
        threading.Thread(target=retire_when_closed, daemon=True).start()
        if not args.no_browser:
            webbrowser.open(server.origin + '/#' + server.token)
        server.serve_forever(poll_interval=.2)
    finally:
        if server:
            server.server_close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / 'startup_error.txt').write_text(str(exc), encoding='utf-8')
        if os.name == 'nt':
            ctypes.windll.user32.MessageBoxW(None, str(exc), '每周预测 · 启动失败', 16)
        raise
