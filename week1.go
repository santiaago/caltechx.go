package main

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// r is a random variable used to generate random points through the program.
var r *rand.Rand

type interval struct {
	min int
	max int
}

// randFloat returns a random float number in the given interval.
func (v *interval) randFloat() float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := float64(v.max - v.min)
	return r.Float64()*size + float64(v.min)
}

// linearsVars holds the variables that define a linear function.
type linearVars struct {
	a float64 // slope
	b float64 // intercept
}

// randLinearVars returns a linearVars struct which holds the random variables of the linear function.
func randLinearVars(i interval) linearVars {
	return linearVars{i.randFloat(), i.randFloat()}
}

// linearFunc is a linear function that takes a float x and returns y = ax + b.
// f(x) = y
type linearFunc func(x float64) float64

// Func returns a linear function with respect of the defined linearVars.
// f(x) = ax + b
// With a and b defined by linearVars.
func (l *linearVars) Func() linearFunc {
	return func(x float64) float64 {
		return x*l.a + l.b
	}
}

// Print will print the linear variables in the following form:
// func: aX + b
func (l *linearVars) Print() {
	fmt.Printf("func: %4.2fX + %4.2f\n", l.a, l.b)
}

// point is a 2 dimentional coordinate (x1 x2).
// with an additional x0 coordinate for the perceptron algorithm.
type point [3]float64

// print will print the coordinates of pt in the following format:
// point: x0:%4.2f \tx1:%4.2f \tx2:%4.2f\n
func (pt *point) print(name string) {
	p := *pt
	fmt.Printf("\t%s0: %4.2f \t%s1: %4.2f \t%s2: %4.2f\t", name, p[0], name, p[1], name, p[2])
}

// evaluate will compare the function f in point p with respect to the current y point.
// if it stands on one side it returns +1 else returns -1
func evaluate(f linearFunc, p point) int {
	if p[2] < f(p[1]) {
		return -1
	}
	return 1
}

// sign returns 1 if number is > than 0 and -1 otherwise
func sign(p float64) int {
	if p > float64(0) {
		return 1
	}
	return -1
}

// Hypothesis function h is the hypothesis of the perceptron algorithm.
// It takes as arguments vector X and a vector W (w for weight)
// function h implements h(x) = sign(w'x)
func h(x point, w point) int {
	if len(x) != len(w) {
		fmt.Println("Panic: vectors x and w should be of same size.")
		panic(x)
	}
	var res float64 = 0
	for i := 0; i < len(w); i++ {
		res = res + w[i]*x[i]
	}
	return sign(res)
}

// extract a misclassified set of indexes from the training set, output and weight vector.
func (pla *PLA) extractMisclassifiedIndexes() []int {
	var set []int
	for i := 0; i < len(pla.Xn); i++ {
		if pla.H(pla.Xn[i], pla.Wn) != pla.Yn[i] {
			set = append(set, i)
		}
	}
	return set
}

// PLA holds all the information needed to run the PLA algorithm.
type PLA struct {
	N              int                        // number of training points
	Interval       interval                   // interval  in which the points, outputs and function are defined.
	RandLinearVars linearVars                 // random vars of random linear function
	Xn             []point                    // data set of random points (uniformly in interval)
	Yn             []int                      // output, evaluation of each Xn based on linear function defined by RandLinearVars
	Wn             point                      // weight vector initialized at zeros.
	H              func(x point, w point) int // hypothesis function.
}

// NewPLA is a constructor of a basic PLA:
// N = 10
// Interval [-1 : 1]
// H = h(x) = sign(w'x)
func NewPLA() *PLA {
	pla := PLA{}
	pla.N = 10
	pla.Interval = interval{-1, 1}
	pla.H = h
	return &pla
}

// updateWeight will update Wn vector with respect to Yn and Xn
func (pla *PLA) updateWeight(n int) {
	for i := 0; i < len(pla.Wn); i++ {
		pla.Wn[i] = pla.Wn[i] + float64(pla.Yn[n])*pla.Xn[n][i]
	}
}

// randMisclassifiedPoint will pick a point at random from the misclassified set.
func (pla *PLA) randMisclassifiedPoint() (int, error) {
	// build the misclassified set:
	indexes := pla.extractMisclassifiedIndexes()
	if len(indexes) == 0 {
		return 0, errors.New("missclassified set is empty.")
	}
	// pick a misclasified point from the set
	rand_index := r.Intn(len(indexes))
	rand_point := indexes[rand_index]
	return rand_point, nil
}

// initialize will set up the PLA structure with the following:
// - the random linear function
// - vector Xn with X0 at 1 and X1 and X2 random point in the defined input space.
// - vector Yn the output of the random linear function on each point Xi. either -1 or +1  based on the linear function.
// - vector Wn is set to zero.
//
func (pla *PLA) initialize() {

	pla.RandLinearVars = randLinearVars(pla.Interval) // create the random vars of the random linear function
	pla.Xn = make([]point, pla.N)
	pla.Yn = make([]int, pla.N)
	pla.Wn[0] = float64(0)
	pla.Wn[1] = float64(0)
	pla.Wn[2] = float64(0)

	for i := 0; i < pla.N; i++ {
		pla.Xn[i][0] = float64(1)
		pla.Xn[i][1] = pla.Interval.randFloat()
		pla.Xn[i][2] = pla.Interval.randFloat()

		pla.Yn[i] = evaluate(pla.RandLinearVars.Func(), pla.Xn[i])
	}
}

func (pla *PLA) converge() int {
	iterations := 0
	for {
		// pick a misclassified point and update the weight vector accordingly
		if randPoint, err := pla.randMisclassifiedPoint(); err == nil {
			pla.updateWeight(randPoint)
		} else {
			break
		}
		//pla.print()
		iterations++
	}
	return iterations
}

func (pla *PLA) disagreement() float64 {

	outOfSample := 1000
	numError := 0

	for i := 0; i < outOfSample; i++ {
		var oX point
		var oY int
		oX[0] = float64(1)
		oX[1] = pla.Interval.randFloat()
		oX[2] = pla.Interval.randFloat()
		oY = evaluate(pla.RandLinearVars.Func(), oX)
		if pla.H(oX, pla.Wn) != oY {
			numError++
		}
	}
	return float64(numError) / float64(outOfSample)
}

// print will display the current random function and the current data hold by vectors Xn, Yn and Wn.
func (pla *PLA) print() {
	pla.RandLinearVars.Print()
	for i := 0; i < pla.N; i++ {
		pla.Xn[i].print("X")
		fmt.Printf("\t Y: %v\n", pla.Yn[i])
	}
	fmt.Println()
	pla.Wn.print("W")
	fmt.Println()
}

// Take N = 10. How many iterations does it take on average for the PLA to
// converge for N = 10 training points?
func q7() {

	avgIterations := 0            // average of iterations that it takes for the PLA to converge
	avgDisagreement := float64(0) // average of disagreement between g (h function with Wn vector) and f
	runs := 1000                  // number of times we repeat the experiment
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	pla := NewPLA()

	for run := 0; run < runs; run++ {
		pla.initialize()
		iterations := pla.converge()
		avgDisagreement += pla.disagreement()
		avgIterations += iterations
	}
	avgIterations = avgIterations / runs
	avgDisagreement = avgDisagreement / float64(runs)
	fmt.Printf("average for PLA to converge for N = 10 is %v\n", avgIterations)
	fmt.Printf("average of disagreement between the hypothesis function and the random function is %4.2f\n", avgDisagreement)
}

func q9() {
	avgIterations := 0            // average of iterations that it takes for the PLA to converge
	avgDisagreement := float64(0) // average of disagreement between g (h function with Wn vector) and f
	runs := 1000                  // number of times we repeat the experiment
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	pla := NewPLA()
	for run := 0; run < runs; run++ {
		pla.N = 100
		pla.initialize()
		iterations := pla.converge()
		avgDisagreement += pla.disagreement()
		avgIterations += iterations
	}
	avgIterations = avgIterations / runs
	avgDisagreement = avgDisagreement / float64(runs)
	fmt.Printf("average for PLA to converge for N = %v is %v\n", pla.N, avgIterations)
	fmt.Printf("average of disagreement between the hypothesis function and the random function for N = %v is %4.2f\n", pla.N, avgDisagreement)

}

func main() {
	fmt.Println("week 1")
	fmt.Println("1 d")
	fmt.Println("2 a")
	fmt.Println("3 e")
	fmt.Println("4 b")
	fmt.Println("5 c")
	fmt.Println("6 e")
	// perceptron learning algorithm
	q7()
	fmt.Println("7 b")
	fmt.Println("8 c")
	q9()
	fmt.Println("9 b")
	fmt.Println("10 b")
}
