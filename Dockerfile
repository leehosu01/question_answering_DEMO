FROM python:3.9
WORKDIR /app
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install --upgrade --force-reinstall  -r requirements.txt
COPY ./app /app
EXPOSE 8080
CMD streamlit run ./app.py --server.port 8080 --server.address 0.0.0.0