@echo off
chcp 65001
cls
echo ===================================================
echo  다빈치 코드 AI (ONNX 버전) 서버 시작
echo ===================================================
echo.
echo Python 웹 서버를 이용하여 로컬 서버를 띄웁니다.
echo 잠시 후 브라우저가 자동으로 열립니다.
echo.
echo [주의] 게임을 하는 동안 이 검은색 창을 끄지 마세요!
echo.

:: 브라우저 실행 (서버가 뜰 때까지 잠시 대기 후 실행할 수도 있겠지만, 보통 동시에 실행해도 브라우저가 재시도하거나 서버가 금방 뜸)
:: 2초 정도 대기 후 브라우저 오픈 (Python 서버 초기화 시간 고려)
timeout /t 2 >nul
start http://localhost:8000/davinci_code_onnx.html

:: 파이썬 서버 실행 (보안 헤더 포함된 전용 서버)
echo 서버 실행 중... (http://localhost:8000)
python server.py

:: 파이썬 실행 실패 시 에러 메시지
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] Python 서버 실행에 실패했습니다.
    echo 1. Python이 설치되어 있는지 확인해주세요. (명령 프롬프트에 'python --version' 입력)
    echo 2. 포트 8000번이 이미 사용 중인지 확인해주세요.
    echo.
    pause
)
