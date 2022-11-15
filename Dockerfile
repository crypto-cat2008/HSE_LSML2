FROM pytorch/pytorch

RUN pip install numpy
RUN pip install pandas
RUN pip install regex
RUN pip install flask
RUN pip install transformers

WORKDIR /myapp
COPY . .

#COPY app.py app.py
#COPY /1111model1 /1111model1

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

ENTRYPOINT ["flask", "run"])