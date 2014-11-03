package biasAndVariance

import (
	"github.com/santiaago/caltechx.go/linear"
	"math"
)

type GFunc func(x float64) float64

type BiasAndVariance struct {
	Interval            linear.Interval   // interval in which the target function is defined.
	TargetFunction      linear.LinearFunc // target function.
	TrainingExampleSize int               // number of training examples of the training set.
	Runs                int               // number of runs that should be used to learn.
	Bias                float64
	Variance            float64
	Slope               float64
}

func NewBiasAndVariance() *BiasAndVariance {
	bav := BiasAndVariance{}
	bav.Interval = linear.Interval{-1, 1}
	bav.Runs = 1000
	bav.TargetFunction = func(x ...float64) float64 {
		return math.Sin(math.Pi * x[0])
	}
	bav.TrainingExampleSize = 2
	return &bav
}

func (bav *BiasAndVariance) LearnLine() {
	sumA := float64(0)

	type GFunc func(x float64, a float64) float64
	gs := make([]func(x float64, a float64) float64, bav.Runs)
	as := make([]float64, bav.Runs)
	for i := 0; i < bav.Runs; i++ {
		x1 := bav.Interval.RandFloat()
		y1 := bav.TargetFunction(x1)

		x2 := bav.Interval.RandFloat()
		y2 := bav.TargetFunction(x2)

		x := math.Abs(x2 - x1)
		y := math.Abs(y2 - y1)

		a := y / x

		sumA += a

		g := func(ix float64, ia float64) float64 { return ia * ix }
		gs[i] = g
		as[i] = a
	}

	bav.Slope = sumA / float64(bav.Runs)

	gBar := func(x float64) float64 {
		return bav.Slope * x
	}

	bav.Bias = bav.ComputeBias(gBar)
	bav.Variance = bav.ComputeVariance(gBar, gs, as)
}

func (bav *BiasAndVariance) ComputeBias(gBar func(x float64) float64) float64 {
	sumBias := float64(0)
	for i := 0; i < bav.Runs; i++ {
		x := bav.Interval.RandFloat()
		sumBias += math.Pow(gBar(x)-bav.TargetFunction(x), float64(2))
	}
	return sumBias / float64(bav.Runs)
}

func (bav *BiasAndVariance) ComputeVariance(gBar func(x float64) float64, gs []func(x float64, a float64) float64, as []float64) float64 {
	sumVar := float64(0)
	for i := 0; i < bav.Runs; i++ {
		for j := 0; j < bav.Runs; j++ {
			x := bav.Interval.RandFloat()
			sumVar += math.Pow(gs[j](x, as[j])-gBar(x), float64(2))
		}
	}
	return sumVar / math.Pow(float64(bav.Runs), float64(2))

}
