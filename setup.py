from setuptools import setup, find_packages

setup(
    name="dynamic_tuna",
    version="0.1.0",  # Update this version as needed
    description="A description of what dynamic_tuna does",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/dynamic_tuna",  # GitHub repo URL
    packages=find_packages(),
    install_requires=[  # Add dependencies if any
        # "example_package>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)