package main

import(
	"fmt"
	"dataworm"
	"flag"
	"runtime/pprof"
	"os"
	"log"
)

func main(){
	train_path := flag.String("train", "train.tsv", "path of training file")
	test_path := flag.String("test", "test.tsv", "path of testing file")
	lambda1 := flag.Float64("lambda1", 0.1, "lambda1")
	lambda2 := flag.Float64("lambda2", 0.1, "lambda2")
	alpha := flag.Float64("alpha", 0.5, "alpha")
	beta := flag.Float64("beta", 1.0, "beta")
	steps := flag.Int("steps", 1, "steps before convergent")
	global_bias_feature_id := flag.Int64("global", -1, "feature id of global bias")
	profile := flag.String("profile", "", "cpu profile filename")
	
	flag.Parse()
	
	if *profile != "" {
        f, err := os.Create(*profile)
        if err != nil {
            log.Fatal(err)
        }
        pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
    }
	
	fmt.Println(*train_path)
	fmt.Println(*test_path)
	
	params := dataworm.FTRLLogisticRegressionParams{
		Alpha: *alpha,
		Beta: *beta,
		Lambda1: *lambda1,
		Lambda2: *lambda2,
		Steps: *steps,
		GlobalBiasFeatureId: *global_bias_feature_id,
	}
	fmt.Println(params)
	auc, _ := dataworm.FTRLLogisticRegression(*train_path, *test_path, params)
	fmt.Println("AUC:")
	fmt.Println(auc)
}