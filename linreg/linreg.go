package linreg

import (
	"fmt"
	"github.com/santiaago/caltechx.go/linear"
)

// PLA holds all the information needed to run the PLA algorithm.
type LinearRegression struct {
	N              int               // number of training points
	Interval       linear.Interval   // interval  in which the points, outputs and function are defined.
	TargetVars     linear.LinearVars // random vars of the random linear function : target function
	TargetFunction linear.LinearFunc // target function
	Xn             []Point           // data set of random points (uniformly in interval)
	Yn             []int             // output, evaluation of each Xn based on linear function defined by RandLinearVars
	Wn             Point             // weight vector initialized at zeros.
}

// NewLinearRegression is a constructor of a basic linear regression structure:
// N = 10
// Interval [-1 : 1]
func NewLinearRegression() *LinearRegression {
	linreg := LinearRegression{}
	linreg.N = 10
	linreg.Interval = linear.Interval{-1, 1}
	return &linreg
}

// Initialize will set up the PLA structure with the following:
// - the random linear function
// - vector Xn with X0 at 1 and X1 and X2 random point in the defined input space.
// - vector Yn the output of the random linear function on each point Xi. either -1 or +1  based on the linear function.
// - vector Wn is set to zero.
func (linreg *LinearRegression) Initialize() {

	linreg.TargetVars = linear.RandLinearVars(linreg.Interval) // create the random vars of the random linear function
	linreg.TargetFunction = linreg.TargetVars.Func()
	linreg.Xn = make([]Point, linreg.N)
	linreg.Yn = make([]int, linreg.N)
	linreg.Wn[0] = float64(0)
	linreg.Wn[1] = float64(0)
	linreg.Wn[2] = float64(0)

	for i := 0; i < linreg.N; i++ {
		linreg.Xn[i][0] = float64(1)
		linreg.Xn[i][1] = linreg.Interval.RandFloat()
		linreg.Xn[i][2] = linreg.Interval.RandFloat()

		linreg.Yn[i] = evaluate(linreg.TargetFunction, linreg.Xn[i])
	}
}

// Learn will compute the pseudo inverse X dager and set W vector accordingly
// Xdager = (X'X)^-1 X'
func (linreg *LinearRegression) Learn() {
	// compute X' <=> X transpose
	XTranspose := make([][]float64, len(linreg.Xn[0]))
	for i := 0; i < len(linreg.Xn[0]); i++ {
		XTranspose[i] = make([]float64, len(linreg.Xn))
	}

	//var XTranspose [len(linreg.Xn[0])][len(linreg.Xn)]float64
	for i := 0; i < len(XTranspose); i++ {
		for j := 0; j < len(XTranspose[0]); j++ {
			XTranspose[i][j] = linreg.Xn[j][i]
		}
	}
	// compute the product of X' and X
	XProduct := make([][]float64, len(linreg.Xn[0]))
	for i := 0; i < len(linreg.Xn[0]); i++ {
		XProduct[i] = make([]float64, len(linreg.Xn[0]))
	}
	//var XProduct [len(linreg.Xn[0])][len(linreg.Xn[0])]float64
	for k := 0; k < len(linreg.Xn[0]); k++ {
		for i := 0; i < len(XTranspose); i++ {
			for j := 0; j < len(XTranspose[0]); j++ {
				XProduct[i][k] += XTranspose[i][j] * linreg.Xn[j][k]
			}
		}
	}
	// inverse XProduct
	mXin := matrix(XProduct)
	Xinv := mXin.inverse()
	// compute product: (X'X)^-1 X'
	XDagger := make([][]float64, len(XProduct))
	for i := 0; i < len(XProduct); i++ {
		XDagger[i] = make([]float64, len(XTranspose[0]))
	}
	for k := 0; k < len(XTranspose[0]); k++ {
		for i := 0; i < len(Xinv); i++ {
			for j := 0; j < len(Xinv[0]); j++ {
				XDagger[i][k] += Xinv[i][j] * XTranspose[j][k]
			}
		}
	}
	linreg.setWeight(matrix(XDagger))
}

func (linreg *LinearRegression) setWeight(d matrix) {

	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[0]); j++ {
			linreg.Wn[i] += d[i][j] * float64(linreg.Yn[j])
		}
	}
}

// sign returns 1 if number is > than 0 and -1 otherwise
func sign(p float64) int {
	if p > float64(0) {
		return 1
	}
	return -1
}

// Ein is the fraction of in sample points which got misclassified.
func (linreg *LinearRegression) Ein() float64 {
	// XnWn
	gInSample := make([]int, len(linreg.Xn))
	for i := 0; i < len(linreg.Xn); i++ {
		gi := float64(0)
		for j := 0; j < len(linreg.Xn[0]); j++ {
			gi += linreg.Xn[i][j] * linreg.Wn[j]
		}
		gInSample[i] = sign(gi)
	}
	nEin := 0
	for i := 0; i < len(gInSample); i++ {
		if gInSample[i] != linreg.Yn[i] {
			nEin++
		}
	}
	return float64(nEin) / float64(len(gInSample))

}

// Eout is the fraction of out of sample points which got misclassified.
func (linreg *LinearRegression) Eout() float64 {
	outOfSample := 1000
	numError := 0

	for i := 0; i < outOfSample; i++ {
		var oX Point
		var oY int
		oX[0] = float64(1)
		oX[1] = linreg.Interval.RandFloat()
		oX[2] = linreg.Interval.RandFloat()
		oY = evaluate(linreg.TargetFunction, oX)
		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if sign(gi) != oY {
			numError++
		}
	}
	return float64(numError) / float64(outOfSample)

}

// Point is a 2 dimentional coordinate (x1 x2).
// Plus a bias coordinate x0 = 1
type Point [3]float64

// print will print the coordinates of pt in the following format:
// point: x0:%4.2f \tx1:%4.2f \tx2:%4.2f\n
func (pt *Point) print(name string) {
	p := *pt
	fmt.Printf("\t%s0: %4.2f \t%s1: %4.2f \t%s2: %4.2f\t\n", name, p[0], name, p[1], name, p[2])
}

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
func evaluate(f linear.LinearFunc, p Point) int {
	if p[2] < f(p[1]) {
		return -1
	}
	return 1
}

// print will display the current random function and the current data hold by vectors Xn, Yn and Wn.
func (linreg *LinearRegression) print() {
	linreg.TargetVars.Print()
	for i := 0; i < linreg.N; i++ {
		//linreg.Xn[i].print("X")
		fmt.Printf("\t Y: %v\n", linreg.Yn[i])
	}
	fmt.Println()
	//linreg.Wn.print("W")
	fmt.Println()
}
