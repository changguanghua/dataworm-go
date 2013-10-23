package dataworm

import (
	"testing"
)

func TestLogisticRegressionPredict(t *testing.T){
	f1 := Feature{Id: 1, Value: 0.1}
	f2 := Feature{Id: 2, Value: 0.3}
	sample := Sample{Features: []Feature{f1, f2}, Label: 1}
	
	model := make(map[int64]float64)
	model[1] = 0.1
	model[2] = 0.1
	
	prediction := LogisticRegressionPredict(sample, model)
	
	if prediction <= 0.5{
		t.Error("prediction should > 0.5")
	}
	
	f1 = Feature{Id: 3, Value: 0.1}
	f2 = Feature{Id: 4, Value: 0.2}
	prediction = LogisticRegressionPredict(Sample{Features: []Feature{f1, f2}, Label: 1}, model)
	if prediction != 0.5 {
		t.Error("prediction should == 0.5")
	}
	
	f1 = Feature{Id: 1, Value: -0.1}
	f2 = Feature{Id: 2, Value: -0.1}
	prediction = LogisticRegressionPredict(Sample{Features: []Feature{f1, f2}, Label: 1}, model)
	if prediction >= 0.5 {
		t.Error("prediction should < 0.5")
	}
}