package linreg

import (
	"fmt"
)

// PLA holds all the information needed to run the PLA algorithm.
type LinearRegression struct {
	N              int                        // number of training points
	Interval       interval                   // interval  in which the points, outputs and function are defined.
	TargetVars     linearVars                 // random vars of the random linear function : target function
	TargetFunction linearFunc                 // target function
	Xn             []point                    // data set of random points (uniformly in interval)
	Yn             []int                      // output, evaluation of each Xn based on linear function defined by RandLinearVars
	Wn             point                      // weight vector initialized at zeros.
	H              func(x point, w point) int // hypothesis function.
}

// NewPLA is a constructor of a basic PLA:
// N = 10
// Interval [-1 : 1]
// H = h(x) = sign(w'x)
func NewLinearRegression() *LinearRegression {
	linreg := LinearRegression{}
	linreg.N = 10
	linreg.Interval = interval{-1, 1}
	linreg.H = h
	return &linreg
}

// initialize will set up the PLA structure with the following:
// - the random linear function
// - vector Xn with X0 at 1 and X1 and X2 random point in the defined input space.
// - vector Yn the output of the random linear function on each point Xi. either -1 or +1  based on the linear function.
// - vector Wn is set to zero.
//
func (linreg *LinearRegression) Initialize() {

	linreg.TargetVars = randLinearVars(linreg.Interval) // create the random vars of the random linear function
	linreg.TargetFunction = linreg.TargetVars.Func()
	linreg.Xn = make([]point, linreg.N)
	linreg.Yn = make([]int, linreg.N)
	linreg.Wn[0] = float64(0)
	linreg.Wn[1] = float64(0)
	linreg.Wn[2] = float64(0)

	for i := 0; i < linreg.N; i++ {
		linreg.Xn[i][0] = float64(1)
		linreg.Xn[i][1] = linreg.Interval.randFloat()
		linreg.Xn[i][2] = linreg.Interval.randFloat()

		linreg.Yn[i] = evaluate(linreg.TargetFunction, linreg.Xn[i])
	}
}

func (linreg *LinearRegression) Learn() {}
func (linreg *LinearRegression) Ein()   {}
func (linreg *LinearRegression) Eout()  {}

// print will display the current random function and the current data hold by vectors Xn, Yn and Wn.
func (linreg *LinearRegression) print() {
	linreg.TargetVars.Print()
	for i := 0; i < linreg.N; i++ {
		linreg.Xn[i].print("X")
		fmt.Printf("\t Y: %v\n", linreg.Yn[i])
	}
	fmt.Println()
	linreg.Wn.print("W")
	fmt.Println()
}
