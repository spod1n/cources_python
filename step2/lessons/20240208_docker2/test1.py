import os
import subprocess

subprocess.run(["docker", "build", "-t", "dck12", "."])
subprocess.run(["docker", "run", "--rm", "-v",  f"{os.getcwd()}:/app", "dck12", "python", "test-op.py"])
