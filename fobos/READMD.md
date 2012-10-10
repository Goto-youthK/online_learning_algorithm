Online machine learning algorithm:FOBOS(Forward-Backward Splitting)
- SVM + L1 regularization
- Logistic regression + L1 regularization

Usage
- make clean
- make

Example
- wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
- bzip2 -d news20.binary.bz2
- ruby shuffle news20.binary > news20.txt
- head -n 15000 news20.txt > news20_train.txt; tail -n 4996 news20.txt > news20_test.txt
- ./main --train news20_train.txt --test news20_test.txt # default is svm
- ./main --train news20_train.txt --test news20_test.txt -c logistic
