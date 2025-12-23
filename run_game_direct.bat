@echo off
chcp 65001
cls
echo ===================================================
echo  다빈치 코드 바로 실행 (서버 없음)
echo ===================================================
echo.
echo 별도의 서버 프로그램 없이 게임을 바로 실행합니다.
echo 게임 전용 크롬 창이 새로 열립니다.
echo.

set "CHROME_EXE=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME_EXE%" (
    echo [오류] 크롬 브라우저를 찾을 수 없습니다.
    echo 구글 크롬이 설치되어 있어야 합니다.
    echo.
    echo 만약 크롬이 설치되어 있다면, 이 파일을 메모장으로 열어
    echo CHROME_EXE 경로를 수정해주세요.
    pause
    exit
)

echo 크롬을 실행합니다...
start "" "%CHROME_EXE%" --allow-file-access-from-files --user-data-dir="%TEMP%\DavinciCodeGameProfile_%RANDOM%" "%~dp0davinci_code.html"
