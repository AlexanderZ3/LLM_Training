@echo off
REM 命令等级：可直接执行 —— 双击本文件即可整夜下载全部数据集与奖励模型。
REM 全部逻辑在 lab\scripts\download_datasets.py 里；本文件只负责用对解释器。
REM 中断后重新双击即可续传，已下好的文件会跳过。
chcp 65001 >nul
setlocal
set "PY=D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
cd /d "%~dp0"

if not exist "%PY%" (
    echo 找不到 conda 解释器：%PY%
    echo 请改本文件里的 PY 变量，指向 ResearchAgentPy310 环境的 python.exe
    pause
    exit /b 2
)

echo 使用解释器：%PY%
echo 目标目录：%~dp0datasets
echo.
"%PY%" "lab\scripts\download_datasets.py" --overnight %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
    echo 全部完成。清单见 datasets\DOWNLOAD_MANIFEST.json
) else (
    echo 退出码 %RC% —— 还有文件未就绪，重新双击本文件可续传。
)
pause
exit /b %RC%
