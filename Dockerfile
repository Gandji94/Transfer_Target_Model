FROM python:3.11

#install system dependencies if needed
#RUN apt-get update && apt-get install -y ...

#workdir inside container
WORKDIR /app

#copy all project files into container
#everything from the main directory will be improted
COPY . .

#install Python dependencies => all required libraries will be imported
RUN pip install --no-cache-dir -r requirements.txt

#running every step, data gathering/cleaning, training and predictiong
CMD ["python", "-m", "src.main", "full-run"]

##building the docker image
#docker build -t transfer-target .

##how to run the container on docker image
#docker run --rm transfer-target