package pla

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// PLA holds all the information needed to run the PLA algorithm.
type PLA struct {
	N              int                        // number of training points
	Interval       interval                   // interval  in which the points, outputs and function are defined.
	TargetVars     linearVars                 // random vars of the random linear function : target function
	TargetFunction linearFunc                 // target function
	Xn             []point                    // data set of random points (uniformly in interval)
	Yn             []int                      // output, evaluation of each Xn based on linear function defined by RandLinearVars
	Wn             point                      // weight vector initialized at zeros.
	H              func(x point, w point) int // hypothesis function.
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

// initialize will set up the PLA structure with the following:
// - the random linear function
// - vector Xn with X0 at 1 and X1 and X2 random point in the defined input space.
// - vector Yn the output of the random linear function on each point Xi. either -1 or +1  based on the linear function.
// - vector Wn is set to zero.
//
func (pla *PLA) Initialize() {

	pla.TargetVars = randLinearVars(pla.Interval) // create the random vars of the random linear function
	pla.TargetFunction = pla.TargetVars.Func()
	pla.Xn = make([]point, pla.N)
	pla.Yn = make([]int, pla.N)
	pla.Wn[0] = float64(0)
	pla.Wn[1] = float64(0)
	pla.Wn[2] = float64(0)

	for i := 0; i < pla.N; i++ {
		pla.Xn[i][0] = float64(1)
		pla.Xn[i][1] = pla.Interval.randFloat()
		pla.Xn[i][2] = pla.Interval.randFloat()

		pla.Yn[i] = evaluate(pla.TargetFunction, pla.Xn[i])
	}
}

// updateWeight will update Wn vector with respect to Yn and Xn
func (pla *PLA) updateWeight(n int) {
	for i := 0; i < len(pla.Wn); i++ {
		pla.Wn[i] = pla.Wn[i] + float64(pla.Yn[n])*pla.Xn[n][i]
	}
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

// randMisclassifiedPoint will pick a point at random from the misclassified set.
func (pla *PLA) randMisclassifiedPoint() (int, error) {
	// build the misclassified set:
	indexes := pla.extractMisclassifiedIndexes()
	if len(indexes) == 0 {
		return 0, errors.New("missclassified set is empty.")
	}
	// pick a misclasified point from the set
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rand_index := r.Intn(len(indexes))
	rand_point := indexes[rand_index]
	return rand_point, nil
}

// Converge will run the PLA algorithm:
// 1 - pick a misclassified point
// 2 - update the weight vector accordingly
// stop when no more misclassified points.
// Returns the number of iterations needed to converge.
func (pla *PLA) Converge() int {
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

// Disagreement will measure the out of sample error of the g function.
// The mesurment is done by generating 1000 out of sample data points and comparing the
// target function and the 'g' (learned) function.
// Return the fraction of error of the learned function with respect to the target function.
func (pla *PLA) Disagreement() float64 {

	outOfSample := 1000
	numError := 0

	for i := 0; i < outOfSample; i++ {
		var oX point
		var oY int
		oX[0] = float64(1)
		oX[1] = pla.Interval.randFloat()
		oX[2] = pla.Interval.randFloat()
		oY = evaluate(pla.TargetFunction, oX)
		if pla.H(oX, pla.Wn) != oY {
			numError++
		}
	}
	return float64(numError) / float64(outOfSample)
}

// print will display the current random function and the current data hold by vectors Xn, Yn and Wn.
func (pla *PLA) print() {
	pla.TargetVars.Print()
	for i := 0; i < pla.N; i++ {
		pla.Xn[i].print("X")
		fmt.Printf("\t Y: %v\n", pla.Yn[i])
	}
	fmt.Println()
	pla.Wn.print("W")
	fmt.Println()
}
