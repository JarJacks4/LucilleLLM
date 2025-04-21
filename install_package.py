import subprocess
import sys

# List of packages to install
packages = ["langchain", "langchain_community", "faiss-cpu", "sentence_transformers", "transformers", "unstructured",
            "datasets","openai","accelerate","bitsandbytes","pymupdf","fastapi","uvicorn"]

# Function to install a package using pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install each package in the list
for package in packages:
    try:
        install_package(package)
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
