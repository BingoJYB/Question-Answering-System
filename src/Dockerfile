# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install necessary tools
RUN apt update
RUN yes | apt install python3-pip
RUN pip3 install nltk gensim

# Run app.py when the container launches
CMD ["python", "system.py", "-OPTIONAL_FLAG"]
