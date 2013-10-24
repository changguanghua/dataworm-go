package main

import(
	"fmt"
	"dataworm"
	"flag"
)

func main(){
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	
	flag.Parse()
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	
	dataworm.RandomDecisionTree(*train_path, *test_path)
}