package hoeffding

import (
	"math"
)

type HoeffdingInequality struct {
	M     int     // number of hypothesis
	N     int     // number of examples
	E     float64 // epsilon
	Bound float64 // bound for 2Mexp(-2(E^2)N)
}

func NewHoeffdingExperiment() HoeffdingInequality {
	h := HoeffdingInequality{}
	h.E = 0.05
	h.Bound = 0.03
	return h
}

func (hoef *HoeffdingInequality) Compute() float64 {
	return float64(2) * float64(hoef.M) * math.Exp(float64(-2)*hoef.E*hoef.E*float64(hoef.N))
}
