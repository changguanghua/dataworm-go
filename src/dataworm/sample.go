package dataworm

type Sample struct {
	Features []Feature
	Label int16
}

func (s *Sample) LabelDoubleValue() float64 {
	return float64(s.Label)
}

type MapBasedSample struct {
	Features map[int64]float64
	Label int16	
}

func (s *MapBasedSample) LabelDoubleValue() float64{
	return float64(s.Label)
}
