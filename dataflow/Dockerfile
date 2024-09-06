FROM apache/beam_python3.9_sdk:2.43.0

WORKDIR /root

# Prebuilt other dependencies
RUN apt-get update
# RUN apt-get install -y libfontconfig1 libgl1-mesa-glx libgl1-mesa-dri

# Set the entrypoint to the Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]

# Install requirements
# This will be cached unless requirements.txt changes, making subsequent builds faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U pip setuptools

# Install main package
COPY pyproject.toml .
COPY src/ src/
RUN pip install .