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
	steps := flag.Int("steps", 1, "steps before convergent")
	global_bias_feature_id := flag.Int64("global", -1, "feature id of global bias")
	
	flag.Parse()
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	
	params := dataworm.LogisticRegressionParams{
		LearningRate: *learning_rate,
		Regularization: *regularization,
		Steps: *steps,
		GlobalBiasFeatureId: *global_bias_feature_id,
	}
	fmt.Println(params)
	auc, _ := dataworm.LogisticRegression(*train_path, *test_path, params)
	fmt.Println("AUC:")
	fmt.Println(auc)
}