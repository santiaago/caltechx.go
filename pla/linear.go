package pla

import (
	"fmt"
	"math/rand"
	"time"
)

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

// evaluate will map function f in point p with respect to the current y point.
// if it stands on one side it is +1 else -1
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
