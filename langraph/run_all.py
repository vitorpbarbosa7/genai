import subprocess, sys

# sobe backend
backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend:fastapi_app", "--port", "8000"])
try:
    # sobe frontend
    subprocess.run(["streamlit", "run", "frontend.py"])
finally:
    backend.terminate()

