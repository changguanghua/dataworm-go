dataworm-go
===========

Golang based machine learning lib


Currently, we support 3 algorithms to do binary classification

We follow SVM data format

* SGD based Logistic Regression

cd src
go build lr.go
./lr --train train.tsv --test test.tsv --learning-rate 0.01 --regularization 0.01 --steps 10

* FTRL based Logistic Regression

cd src
go build ftrl.go
./ftrl --train train.tsv --test test.tsv

* Random Decision Tree

cd src
go build rdt.go
./rdt --train train.tsv --test test.tsv --tree-count 10 --min-leaf-size 10

