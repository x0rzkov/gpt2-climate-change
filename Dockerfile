FROM python:3.7.6-slim-stretch

RUN apt-get update && \
    apt-get install -y nano locate bash
    
WORKDIR /opt/app
COPY . /opt/app

RUN pip install -U pip && \
    # pip install virtualenv && \
    # virtualenv .venv && \
    # source .venv/bin/activate && \
    pip3 install -r requirements.txt

CMD ["/bin/bash"]


