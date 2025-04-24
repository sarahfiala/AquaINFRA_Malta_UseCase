# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all local files into the container's /app folder
COPY . /app

# Install required Python packages (if any)
RUN pip install flopy pyvista xarray pandas matplotlib pvgeo || true

# Default command: run your setup script
CMD ["python", "src/setup_model.py"]
