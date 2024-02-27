#Base Image
FROM python:3.11-slim

#Set the working directory in the container
WORKDIR /app

#Copy the current directory contents into the container at /app
COPY . .

#Install debian/ubuntu packages and update the container
RUN apt-get update && apt-get install -y python3-opencv

#Install the python packages
RUN pip install --no-cache-dir -r requirements.txt

#Run the python script
CMD ["python", "./dataset.py"]
