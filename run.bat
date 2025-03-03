@echo off

echo Installing Python dependencies...
cd adapters\image_processor || (echo Failed to cd into adapters/image_processor & exit /b 1)
pip install -r requirements.txt || (echo Failed to install Python dependencies & exit /b 1)

echo Downloading models...
python setup.py || (echo Failed to download models & exit /b 1)

echo Starting Python AI server...
start cmd /k python main.py > python_server.log 2>&1

echo Waiting for Python to start...
timeout /t 5

echo Starting Go API server...
cd ..\..\adapters\api || (echo Failed to cd into adapters/api & exit /b 1)
go mod tidy || (echo Failed to run go mod tidy & exit /b 1)
go run main.go || (echo Failed to run Go server & exit /b 1)

pause