package dataworm


type LogisticRegressionParams struct {
	LearningRate float64
	Regularization float64
	Steps int
	GlobalBiasFeatureId int64
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

func LogisticRegressionTrain(dataset DataSet, params LogisticRegressionParams) (model map[int64]float64) {
	model = make(map[int64]float64)
	for step := 0; step < params.Steps; step++ {
		for _, sample := range dataset.Samples {
			prediction := LogisticRegressionPredict(sample, model)
			err := sample.LabelDoubleValue() - prediction
			for _, feature := range sample.Features {
				model_feature_value, ok := model[feature.Id]
				if !ok {
					model_feature_value = 0.0
				}
				model_feature_value += params.LearningRate * (err * feature.Value - params.Regularization * model_feature_value)
				model[feature.Id] = model_feature_value
			}
		}
	}
	return model
}

func LogisticRegression(train_path string, test_path string, params LogisticRegressionParams) (auc float64, err error){
	train_dataset := DataSet{}
	err = train_dataset.Load(train_path, params.GlobalBiasFeatureId)
	if err != nil{
		return 0.5, err
	}
	
	test_dataset := DataSet{}
	err = test_dataset.Load(test_path, params.GlobalBiasFeatureId)
	if err != nil{
		return 0.5, err
	}
	
	model := LogisticRegressionTrain(train_dataset, params)
	
	predictions := []LabelPrediction{}
	for _, sample := range test_dataset.Samples {
		prediction := LogisticRegressionPredict(sample, model)
		predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
	}
	
	auc = AUC(predictions)
	return auc, nil
}

