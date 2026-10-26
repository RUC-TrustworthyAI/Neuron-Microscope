"""Host API meter or one stage's tool socket. Started and stopped by run.sh."""
import http.server
import os
import signal
import socketserver
import sys
import threading
import time
from pathlib import Path
from common import read,write


def serve(server):
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:server.serve_forever()
    finally:server.server_close()


def main(args):
    if args[0]=='meter':
        import api_meter
        batch,ready=map(Path,args[1:]);budget=api_meter.Budget(batch)
        Handler=api_meter.handler(budget)
        class MeterServer(http.server.ThreadingHTTPServer):
            daemon_threads=False
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name, self.server_port = self.server_address
        server=MeterServer(('0.0.0.0',0),Handler)
        mask=os.umask(0o077)
        try:write(ready,{'port':server.server_port,'token':budget.token})
        finally:os.umask(mask)
        serve(server)
    elif args[0]=='session':
        from session import ToolServer
        from stage_session import StageSession
        config=read(args[1]);config['_deadline']=time.monotonic()+int(args[3])
        server=ToolServer(Path(args[2]),StageSession(config))
        server.daemon_threads=False
        serve(server)
    else:raise ValueError(args[0])

if __name__=='__main__':main(sys.argv[1:])
