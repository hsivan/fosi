FROM python:3.9

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install /app

EXPOSE 6400
ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR "/app/experiments"
CMD python dnn/logistic_regression_mnist.py