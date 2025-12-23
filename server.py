import http.server
import socketserver
import os

PORT = 8000

class COIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # SharedArrayBuffer 및 WASM 고성능 모드를 위한 보안 헤더 (Cross-Origin Isolation)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

# 현재 디렉토리에서 실행 확인
if __name__ == "__main__":
    # 포트 재사용 허용
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), COIHandler) as httpd:
        print(f"===================================================")
        print(f" 다빈치 코드 AI 전용 서버 실행 중 (Port: {PORT})")
        print(f" 보안 헤더(COOP/COEP)가 적용되었습니다.")
        print(f" 브라우저에서 http://localhost:{PORT}/davinci_code_onnx.html 접속")
        print(f"===================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n서버를 종료합니다.")
