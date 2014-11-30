package logreg

import (
	"fmt"
	"github.com/santiaago/caltechx.go/linear"
	"math"
	"math/rand"
)

type LogisticRegression struct {
	N              int             // number of training points
	D              int             // dimention
	Interval       linear.Interval // interval  in which the points, outputs and function are defined.
	Eta            float64         //learning rate
	Epsilon        float64
	LinearVars     linear.LinearVars // random vars for linear function
	TargetFunction linear.LinearFunc // random linear function
	Xn             [][]float64       // data set of random points (uniformly chosen in interval)
	Yn             []int             // output, evaluation of each Xi based on the linear random function.
	Wn             []float64         // weight vector.
	VectorSize     int               // size of vectors Xi and Wi
	Epochs         int               // number of epochs
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
	logreg.Epsilon = 0.01
	logreg.VectorSize = 3
	return &logreg
}

type LinearFunc func(x float64) float64

func (logreg *LogisticRegression) Initialize() {
	// take two random points and create random function
	x1 := logreg.Interval.RandFloat()
	y1 := logreg.Interval.RandFloat()

	x2 := logreg.Interval.RandFloat()
	y2 := logreg.Interval.RandFloat()

	slope := (y2 - y1) / (x2 - x1)
	itersect := y1 - slope*x1
	logreg.LinearVars.A = slope
	logreg.LinearVars.B = itersect
	logreg.TargetFunction = logreg.LinearVars.Func()

	logreg.Xn = make([][]float64, logreg.N)
	logreg.Wn = make([]float64, logreg.VectorSize)

	for i := 0; i < logreg.N; i++ {
		logreg.Xn[i] = make([]float64, logreg.VectorSize)
	}
	logreg.Yn = make([]int, logreg.N)

	for i := 0; i < logreg.N; i++ {
		logreg.Xn[i][0] = float64(1)
		for j := 1; j < len(logreg.Xn[i]); j++ {
			logreg.Xn[i][j] = logreg.Interval.RandFloat()
		}
		logreg.Yn[i] = evaluate(logreg.TargetFunction, logreg.Xn[i])
	}
}

// use SGD here.
func (logreg *LogisticRegression) Learn() {

	logreg.Epochs = 0
	indexes := buildIndexArray(logreg.N)
	for {
		shuffleArray(&indexes)
		wOld := make([]float64, len(logreg.Wn))
		copy(wOld, logreg.Wn)
		for i := range indexes {
			wi := logreg.Xn[i][1:]
			yi := logreg.Yn[i]
			gt := logreg.Gradient(wi, yi)
			logreg.UpdateWeights(gt)
		}
		logreg.Epochs++
		if logreg.Converged(wOld) {
			break
		}
	}
}

// Returns the gradient vector with respect to:
// the current sample wi
// the current target value:yi
// the current weights: Wn
func (logreg *LogisticRegression) Gradient(wi []float64, yi int) []float64 {
	v := make([]float64, len(wi)+1)
	v[0] = float64(yi)
	for i, x := range wi {
		v[i+1] = float64(yi) * x
	}
	a := make([]float64, len(wi)+1)
	a[0] = float64(1)
	for i, _ := range wi {
		a[i+1] = wi[i]
	}
	b := make([]float64, len(logreg.Wn))
	copy(b, logreg.Wn)
	d := float64(1) + math.Exp(float64(yi)*dot(a, b))

	//vG = [-1.0 * x / d for x in vector]
	vg := make([]float64, len(v))
	for i, _ := range v {
		vg[i] = float64(-1) * v[i] / d
	}
	return vg
}

// UpdateWeights function update the weights given the curren weights Wn the gradient vector gt with respect of the learning rate Eta.
func (logreg *LogisticRegression) UpdateWeights(gt []float64) {

	if len(gt) != len(logreg.Wn) {
		fmt.Println("Panic: length of Wn and gt should be equal")
		panic(gt)
	}

	newW := make([]float64, len(logreg.Wn))
	for i, _ := range logreg.Wn {
		newW[i] = (logreg.Wn[i] - logreg.Eta*gt[i])
	}
	logreg.Wn = newW
}

func (logreg *LogisticRegression) Converged(wOld []float64) bool {
	diff := make([]float64, len(wOld))
	for i, _ := range wOld {
		diff[i] = logreg.Wn[i] - wOld[i]
	}
	return norm(diff) < logreg.Epsilon
}

func norm(v []float64) float64 {
	return math.Sqrt(dot(v, v))
}

func dot(a, b []float64) float64 {
	if len(a) != len(b) {
		fmt.Println("Panic: lenght of a, and b should be equal")
		panic(a)
	}
	var ret float64
	for i, _ := range a {
		ret += a[i] * b[i]
	}
	return ret
}

func buildIndexArray(n int) []int {
	indexes := make([]int, n)
	for i, _ := range indexes {
		indexes[i] = i
	}
	return indexes
}

func shuffleArray(a *[]int) {
	slice := *a
	for i := range slice {
		j := rand.Intn(i + 1)
		slice[i], slice[j] = slice[j], slice[i]
	}
}

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
// todo: might change name to mapPoint
func evaluate(f linear.LinearFunc, p []float64) int {
	if p[2] < f(p[1]) {
		return -1
	}
	return 1
}

// func (lr *LogisticRegression) Ein() float64 {

// }

// Eout is the out of sample error of the logistic regression.
// It uses the cross entropy error given a generated data set and the weight vector Wn
func (logreg *LogisticRegression) Eout() float64 {
	outOfSample := 1000
	cee := float64(0)

	for i := 0; i < outOfSample; i++ {
		var oY int
		oX := make([]float64, logreg.VectorSize)
		oX[0] = float64(1)
		for j := 1; j < len(oX); j++ {
			oX[j] = logreg.Interval.RandFloat()
		}
		oY = evaluate(logreg.TargetFunction, oX)
		cee += logreg.CrossEntropyError(oX, oY)
	}
	return cee / float64(outOfSample)
}

// CrossEntropyError computes the cross entropy error given a sample X and its target,
// with respect to weight vector Wn based on formula:
// log(1 + exp(-y*sample*w))
func (logreg *LogisticRegression) CrossEntropyError(sample []float64, Y int) float64 {
	return math.Log(float64(1) + math.Exp(float64(-Y)*dot(sample, logreg.Wn)))
}
