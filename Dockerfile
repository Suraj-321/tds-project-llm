FROM python:3.12-slim-bookworm

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set up the application directory
WORKDIR /app

# Copy application files
COPY app.py /app
COPY tasksA.py /app
COPY tasksB.py /app

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Expose port
EXPOSE 8000

# Run the application
CMD ["/root/.local/bin/uv", "run", "app.py"]