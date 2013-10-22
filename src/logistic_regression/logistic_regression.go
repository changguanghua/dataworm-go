package main

import "fmt"
import "math"

type Feature struct {
	id int64
	value float64
}

type Sample struct {
	features []Feature
	label int16
}

func (s *Sample) LabelDoubleValue() float64 {
	return float64(s.label)
}

type DataSet struct {
	samples []Sample
}

func (d *DataSet) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
  	for scanner.Scan() {
		
	}
	return scanner.Err()	
}

func Sigmoid(x float64)(y float64) {
	y = 1 / (1 + math.Exp(-1 * x))
	return y
}

func Predict(sample Sample, model map[int64] float64) (ret float64) {
	ret = 0
	for _, feature := range sample.features {
		model_feature_value, ok := model[feature.id]
		if ok {
			ret += model_feature_value * feature.value	
		}
	}
	return Sigmoid(ret)
}

func Train(dataset DataSet) (model map[int64]float64) {
	model = make(map[int64]float64)
	alpha := 0.01
	lambda := 0.01
	for _, sample := range dataset.samples {
		prediction := Predict(sample, model)
		err := sample.LabelDoubleValue() - prediction
		for _, feature := range sample.features {
			model_feature_value, ok := model[feature.id]
			if !ok {
				model_feature_value = 0.0
			}
			model_feature_value += alpha * (err * feature.value - lambda * model_feature_value)
			model[feature.id] = model_feature_value
		}
	}
	return model
}

func main(){
	f1 := Feature{3, 0.1}
	f2 := Feature{4, 0.7}
	
	fmt.Println(f1)
	fmt.Println(f2)
	
	s1 := Sample{features: []Feature{f1, f2}, label: 1}
	fmt.Println(s1)
	
	dataset := DataSet{samples: []Sample{s1}}
	
	Train(dataset)
	
	model := make(map[int64] float64)
	model[3] = 0.9
	model[7] = 0.7
	
	fmt.Println(model)
	
	prediction := Predict(s1, model)
	
	fmt.Println(prediction)
}