```shell

docker build -t leehosu01/question_answering_demo .
docker push leehosu01/question_answering_demo
docker run -p 80:8080 leehosu01/question_answering_demo
```