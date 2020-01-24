import sys
import SimpleHTTPServer
import SocketServer

PORT = 8000

dir = None
if len(sys.argv) > 1:
    dir = sys.argv[1]
    print('Serving directory ', sys.argv[1])


PORT = 8000

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.webapp': 'application/x-web-app-manifest+json',
});

httpd = SocketServer.TCPServer(("", PORT), Handler)

print "Serving at port", PORT
httpd.serve_forever()