from setuptools import setup, find_packages

setup(
    name="rag-package",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "docling>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    author="Alexandra",
    author_email="alexandranac2@gmail.com",
    description="Reusable RAG system with multi-collection support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-package",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
