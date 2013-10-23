package main

import(
	"fmt"
	"dataworm"
	"flag"
)

func main(){
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	learning_rate := flag.Float64("learning-rate", 0.01, "learning rate")
	regularization := flag.Float64("regularization", 0.01, "regularization")
	flag.Parse()
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	auc, _ := dataworm.LogisticRegression(*train_path, *test_path, *learning_rate, *regularization)
	fmt.Println(auc)
}