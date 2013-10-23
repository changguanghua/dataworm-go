package dataworm

import (
	"math"
)

func Sigmoid(x float64)(y float64) {
	y = 1 / (1 + math.Exp(-1 * x))
	return y
}

func LogisticRegressionPredict(sample Sample, model map[int64] float64) (ret float64) {
	ret = 0
	for _, feature := range sample.Features {
		model_feature_value, ok := model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value	
		}
	}
	return Sigmoid(ret)
}

func LogisticRegressionTrain(dataset DataSet, learning_rate float64, regularization float64) (model map[int64]float64) {
	model = make(map[int64]float64)
	for _, sample := range dataset.Samples {
		prediction := LogisticRegressionPredict(sample, model)
		err := sample.LabelDoubleValue() - prediction
		for _, feature := range sample.Features {
			model_feature_value, ok := model[feature.Id]
			if !ok {
				model_feature_value = 0.0
			}
			model_feature_value += learning_rate * (err * feature.Value - regularization * model_feature_value)
			model[feature.Id] = model_feature_value
		}
	}
	return model
}

func LogisticRegression(train_path string, test_path string, learning_rate float64, regularization float64) (auc float64, err error){
	train_dataset := DataSet{}
	err = train_dataset.Load(train_path)
	if err != nil{
		return 0.5, err
	}
	
	test_dataset := DataSet{}
	err = test_dataset.Load(test_path)
	if err != nil{
		return 0.5, err
	}
	
	model := LogisticRegressionTrain(train_dataset, learning_rate, regularization)
	
	predictions := []LabelPrediction{}
	for _, sample := range test_dataset.Samples {
		prediction := LogisticRegressionPredict(sample, model)
		predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
	}
	
	auc = AUC(predictions)
	return auc, nil
}

