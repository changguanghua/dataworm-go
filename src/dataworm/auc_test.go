package dataworm

import (
	"testing"
	"math/rand"
	"math"
)

func TestAUC(t *testing.T){
	predictions := []LabelPrediction{}
	for i := 0; i < 1000; i++ {
		predictions = append(predictions, LabelPrediction{Label: int16(rand.Int() % 2), Prediction: rand.Float64()})
	}
	auc := AUC(predictions)
	if math.Abs(auc - 0.5) > 0.05{
		t.Error("Random predictions should have auc arround 0.5")
	}
	
	predictions = nil
	for i := 0; i < 1000; i++ {
		label := int16(rand.Int() % 2)
		prediction := rand.Float64()
		if label == 1 {
			prediction += 1.0
		}
		predictions = append(predictions, LabelPrediction{Label: label, Prediction: prediction})
	}
	auc = AUC(predictions)
	if auc < 0.6 {
		t.Error("Asending predictions should have auc > 0.5")
	}
}