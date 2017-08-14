FROM mfeurer/auto-sklearn:development

COPY src/ /usr/src/
WORKDIR /usr/src/

CMD ["python3", "main.py"]
