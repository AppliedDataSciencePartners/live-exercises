# Start with Python base image
FROM python3.11.3-slim-buster

# Install dependencies
COPY requirements.txt /requirements
RUN pip3 install -r /requirements/requirements.txt

# Create a non-root user
RUN groupadd -g 2222 appgroup
RUN useradd -m -u 2222 --gid 2222 user

# Copy the source code into the /app directory
COPY src .
RUN chown -R appuser:appgroup /app
USER appuser

# Specify the port number the container should expose
EXPOSE 8000

# Set entrypoint to run the application
CMD ["./run.sh"]