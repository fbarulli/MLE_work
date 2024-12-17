FROM python:3.12

# Install required tools and Jupyter
RUN apt-get update && apt-get install -y wget && \
    pip install --no-cache-dir jupyter notebook

# Expose Jupyter's default port
EXPOSE 8888

# Default command to start Jupyter Notebook server
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]
