echo Downloading Movies
wget -P data/ http://www.eraserbenchmark.com/zipped/movies.tar.gz
tar xvzf data/movies.tar.gz -C data/
rm data/movies.tar.gz
echo Downloading MultiRC
wget -P data/ http://www.eraserbenchmark.com/zipped/multirc.tar.gz
tar xvzf data/multirc.tar.gz -C data/
rm data/multirc.tar.gz
echo Downloading Fever
wget -P data/ http://www.eraserbenchmark.com/zipped/fever.tar.gz
tar xvzf data/fever.tar.gz -C data/
rm data/fever.tar.gz
echo Downloading BoolQ
wget -P data/ http://www.eraserbenchmark.com/zipped/boolq.tar.gz
tar xvzf data/boolq.tar.gz -C data/
rm data/boolq.tar.gz
echo Downloading Evidence-Inference
wget -P data/ http://www.eraserbenchmark.com/zipped/evidence_inference.tar.gz
tar xvzf data/evidence_inference.tar.gz -C data/
rm data/evidence_inference.tar.gz
