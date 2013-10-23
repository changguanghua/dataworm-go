package dataworm

type Sample struct {
	Features []Feature
	Label int16
}

func (s *Sample) LabelDoubleValue() float64 {
	return float64(s.Label)
}