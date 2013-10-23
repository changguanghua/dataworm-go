package dataworm

import (
	"math"
	"fmt"
)

type FTRLLogisticRegressionParams struct {
	Alpha, Beta, Lambda1, Lambda2 float64
	Steps int
	GlobalBiasFeatureId int64
}

type FTRLFeatureWeight struct {
	ni, zi float64
}

func (w *FTRLFeatureWeight) Wi(p FTRLLogisticRegressionParams) float64 {
	wi := 0.0
	if math.Abs(w.zi) > p.Lambda1 {
		wi = (Signum(w.zi) * p.Lambda1 - w.zi) / (p.Lambda2 + (p.Beta + math.Sqrt(w.ni)) / p.Alpha);	
	}
	return wi
}

func FTRLLogisticRegressionPredict(sample Sample, model map[int64]FTRLFeatureWeight, p FTRLLogisticRegressionParams) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := model[feature.Id]
		if ok {
			ret += model_feature_value.Wi(p) * feature.Value	
		}
	}
	return Sigmoid(ret)
}

func FTRLLogisticRegressionTrain(dataset DataSet, params FTRLLogisticRegressionParams) map[int64]float64 {
	model := make(map[int64]FTRLFeatureWeight)
	for step := 0; step < params.Steps; step++ {
		for _, sample := range dataset.Samples {
			prediction := FTRLLogisticRegressionPredict(sample, model, params)
			err := sample.LabelDoubleValue() - prediction
			for _, feature := range sample.Features {
				model_feature_value, ok := model[feature.Id]
				if !ok {
					model_feature_value = FTRLFeatureWeight{0.0, 0.0}
				}
				zi := model_feature_value.zi
				ni := model_feature_value.ni
				gi := -1 * err * feature.Value
				sigma := (math.Sqrt(ni + gi * gi) - math.Sqrt(ni)) / params.Alpha
				wi := model_feature_value.Wi(params)
				zi += gi - sigma * wi
				ni += gi * gi
				model[feature.Id] = FTRLFeatureWeight{zi: zi, ni: ni}
			}
		}
	}
	shrink_model := make(map[int64]float64)
	for id, weight := range model{
		wi := weight.Wi(params)
		if math.Abs(wi) > 1E-7{
			shrink_model[id] = wi
		}
	}
	return shrink_model
}

func FTRLLogisticRegression(train_path string, test_path string, params FTRLLogisticRegressionParams) (auc float64, err error){
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
	
	model := FTRLLogisticRegressionTrain(train_dataset, params)
	
	fmt.Println(len(model))
	
	predictions := []LabelPrediction{}
	for _, sample := range test_dataset.Samples {
		prediction := LogisticRegressionPredict(sample, model)
		predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
	}
	
	auc = AUC(predictions)
	return auc, nil
}

