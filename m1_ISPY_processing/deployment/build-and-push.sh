cd ../ 
docker build . -f deployment/Dockerfile -t  gcr.io/long-loop-273905/m1:latest 
docker push gcr.io/long-loop-273905/m1:latest 
cd -
