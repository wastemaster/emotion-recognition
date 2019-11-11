line=`shuf -n 1 emotional-lines.txt`

curl http://127.0.0.1:5000/api/predictor -d "essay=$line" -X POST -v
