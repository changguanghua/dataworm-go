package main

import(
	"fmt"
	"dataworm"
	"flag"
	"runtime"
)

func main(){
	runtime.GOMAXPROCS(runtime.NumCPU()) 
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	tree_count := flag.Int("tree-count", 10, "number of random trees")
	min_leaf_size := flag.Int("min-leaf-size", 10, "min leaf size")
	flag.Parse()
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	params := dataworm.RDTParams{TreeCount: *tree_count, MinLeafSize: *min_leaf_size}
	fmt.Println(params)
	dataworm.RandomDecisionTree(*train_path, *test_path, params)
}