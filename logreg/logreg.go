package logreg

import (
	"github.com/santiaago/caltechx.go/linear"
	"math"
)

type LogisticRegression struct {
	N              int               // number of training points
	D              int               // dimention
	Interval       linear.Interval   // interval  in which the points, outputs and function are defined.
	Eta            float64           //learning rate
	LinearVars     linear.LinearVars // random vars for linear function
	TargetFunction linear.LinearFunc // random linear function
	Xn             [][]float64       // data set of random points (uniformly chosen in interval)
	Yn             []int             // output, evaluation of each Xi based on the linear random function.
	VectorSize     int               // size of vectors Xi and Wi
}

// NewLogisticRegression is a constructor of a basic logistic regression structure:
// N = 100
// Interval [-1 : 1]
// Learning rate: 0.01
func NewLogisticRegression() *LogisticRegression {
	logreg := LogisticRegression{}
	logreg.N = 100
	logreg.Interval = linear.Interval{-1, 1}
	logreg.Eta = 0.01
	logreg.VectorSize = 3
	return &logreg
}

type LinearFunc func(x float64) float64

func (logreg *LogisticRegression) Initialize() {
	// take two random points and create random function
	x1 := linear.Interval.RandFloat()
	y1 := linear.Interval.RandFloat()

	x2 := linear.Interval.RandFloat()
	y2 := linear.Interval.RandFloat()

	slope := (y2 - y1) / (x2 - x1)
	itersect := y1 - slope*x1
	logreg.LinearVars.A = slope
	logreg.LinearVars.B = itersect
	logreg.TargetFunction = logreg.LinearVars.Func()

	logreg.Xn = make([][]float64, logreg.N)
	for i := 0; i < logreg.N; i++ {
		logreg.Xn[i] = make([]float64, logreg.VectorSize)
	}
	linreg.Yn = make([]int, linreg.N)

	for i := 0; i < logreg.N; i++ {
		logreg.Xn[i][0] = float64(1)
		for j := 1; j < len(logreg.Xn[i]); j++ {
			logreg.Xn[i][j] = logreg.Interval.RandFloat()
		}
		linreg.Yn[i] = evaluate(logreg.TargetFunction, logreg.Xn[i])
	}

}

// use SGD here.
func Learn() {

}

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
// todo: might change name to mapPoint
func evaluate(f logreg.LinearFunc, p []float64) int {
	if p[2] < f(p[1]) {
		return -1
	}
	return 1
}

func (lr *LogisticRegression) Ein() float64 {

}

func (lr *LogisticRegression) Eout() float64 {

}
