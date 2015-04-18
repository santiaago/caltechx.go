package biasAndVariance

import (
	"fmt"
	"math"

	"github.com/santiaago/ml/linear"
)

type GFunc func(x float64) float64

type BiasAndVariance struct {
	Interval            linear.Interval // interval in which the target function is defined.
	TargetFunction      linear.Function // target function.
	TrainingExampleSize int             // number of training examples of the training set.
	Runs                int             // number of runs that should be used to learn.
	Bias                float64
	Variance            float64
	Slope               float64
	Constant            float64
	ThroughOrigin       bool // defines if hypothesis function goes through origin or not. ie: h(x) = ax or h(x) = ax + b
}

func NewBiasAndVariance() *BiasAndVariance {
	bav := BiasAndVariance{}
	bav.Interval = linear.NewInterval(float64(-1), float64(1))
	bav.Runs = 1000
	bav.TargetFunction = func(x float64) float64 {
		return math.Sin(math.Pi * x)
	}
	bav.TrainingExampleSize = 2
	bav.ThroughOrigin = true
	return &bav
}

func (bav *BiasAndVariance) LearnLine() {
	sumA := float64(0)
	sumB := float64(0)

	type GFunc func(x, a float64) float64
	gs := make([]func(x, a, b float64) float64, bav.Runs)
	as := make([]float64, bav.Runs)
	bs := make([]float64, bav.Runs)

	for i := 0; i < bav.Runs; i++ {
		x1 := bav.Interval.RandFloat()
		y1 := bav.TargetFunction(x1)

		x2 := bav.Interval.RandFloat()
		y2 := bav.TargetFunction(x2)

		x := math.Abs(x2 - x1)
		y := math.Abs(y2 - y1)

		a := y / x
		sumA += a
		as[i] = a

		g := func(ix, ia, ib float64) float64 { return ia*ix + ib }
		gs[i] = g

		if !bav.ThroughOrigin {
			b := y1 - a*x1
			sumB += b
			bs[i] = b
		}

	}

	bav.Slope = sumA / float64(bav.Runs)
	bav.Constant = sumB / float64(bav.Runs)

	gBar := func(x float64) float64 {
		return bav.Slope*x + bav.Constant
	}

	bav.Bias = bav.ComputeBias(gBar)
	bav.Variance = bav.ComputeVariance(gBar, gs, as, bs)
}

func (bav *BiasAndVariance) LearnConstant() {
	sumB := float64(0)

	type GFunc func(x float64, a float64) float64
	gs := make([]func(x, a, b float64) float64, bav.Runs)
	bs := make([]float64, bav.Runs)
	for i := 0; i < bav.Runs; i++ {
		x1 := bav.Interval.RandFloat()
		y1 := bav.TargetFunction(x1)

		x2 := bav.Interval.RandFloat()
		y2 := bav.TargetFunction(x2)

		b := (y1 + y2) / float64(2)

		sumB += b

		g := func(ix, ia, ib float64) float64 { return ib }
		gs[i] = g
		bs[i] = b
	}

	bav.Constant = sumB / float64(bav.Runs)

	gBar := func(x float64) float64 {
		return bav.Constant
	}

	bav.Bias = bav.ComputeBias(gBar)
	bav.Variance = bav.ComputeVariance(gBar, gs, make([]float64, bav.Runs), bs)

}

func (bav *BiasAndVariance) LearnQuadratic() {
	sumA := float64(0)
	sumB := float64(0)

	type GFunc func(x float64, a float64) float64
	gs := make([]func(x, a, b float64) float64, bav.Runs)
	as := make([]float64, bav.Runs)
	bs := make([]float64, bav.Runs)

	for i := 0; i < bav.Runs; i++ {
		x1 := bav.Interval.RandFloat()
		y1 := bav.TargetFunction(x1)

		x2 := bav.Interval.RandFloat()
		y2 := bav.TargetFunction(x2)

		a := (y1 - y2) / (x1*x1 - x2*x2)
		as[i] = a

		if !bav.ThroughOrigin {
			b := y1 - a*x1*x1
			bs[i] = b
			sumB += b
		}
		g := func(ix, ia, ib float64) float64 { return ia*ix*ix + ib }
		gs[i] = g
	}

	// this should definitely not be called slope but First Coefficient or something similar.
	bav.Slope = sumA / float64(bav.Runs)

	if !bav.ThroughOrigin {
		bav.Constant = sumB / float64(bav.Runs)
	}

	gBar := func(x float64) float64 {
		return x*x*bav.Slope + bav.Constant
	}

	bav.Bias = bav.ComputeBias(gBar)
	bav.Variance = bav.ComputeVariance(gBar, gs, as, bs)
}

func (bav *BiasAndVariance) ComputeBias(gBar func(x float64) float64) float64 {
	sumBias := float64(0)
	for i := 0; i < bav.Runs; i++ {
		x := bav.Interval.RandFloat()
		sumBias += math.Pow(gBar(x)-bav.TargetFunction(x), float64(2))
	}
	return sumBias / float64(bav.Runs)
}

func (bav *BiasAndVariance) ComputeVariance(gBar func(x float64) float64, gs []func(x, a, b float64) float64, as []float64, bs []float64) float64 {
	sumVar := float64(0)
	for i := 0; i < bav.Runs; i++ {
		for j := 0; j < bav.Runs; j++ {
			x := bav.Interval.RandFloat()
			sumVar += math.Pow(gs[j](x, as[j], bs[j])-gBar(x), float64(2))
		}
	}
	return sumVar / math.Pow(float64(bav.Runs), float64(2))

}

func (bav *BiasAndVariance) Print() {
	fmt.Printf("bias = %3.2f\n", bav.Bias)
	fmt.Printf("variance = %3.2f\n", bav.Variance)
	fmt.Printf("Eout = %3.2f\n", bav.Bias+bav.Variance)
}
