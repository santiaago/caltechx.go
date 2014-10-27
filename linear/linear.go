package linear

import (
	"fmt"
	"math/rand"
	"time"
)

type Interval struct {
	Min int
	Max int
}

// randFloat returns a random float number in the given interval.
func (v *Interval) RandFloat() float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := float64(v.Max - v.Min)
	return r.Float64()*size + float64(v.Min)
}

// LinearsVars holds the variables that define a linear function.
type LinearVars struct {
	A float64 // slope
	B float64 // intercept
}

// randLinearVars returns a LinearVars struct which holds the random variables of the linear function.
func RandLinearVars(i Interval) LinearVars {
	return LinearVars{i.RandFloat(), i.RandFloat()}
}

// LinearFunc is a linear function that takes a float x and returns y = ax + b.
// f(x) = y
type LinearFunc func(x ...float64) float64

// Func returns a linear function with respect of the defined linearVars.
// f(x) = ax + b
// With a and b defined by linearVars.
func (l *LinearVars) Func() LinearFunc {
	return func(x ...float64) float64 {
		return x[0]*l.A + l.B
	}
}

// Print will print the linear variables in the following form:
// func: aX + b
func (l *LinearVars) Print() {
	fmt.Printf("func: %4.2fX + %4.2f\n", l.A, l.B)
}

// // sign returns 1 if number is > than 0 and -1 otherwise
func Sign(p float64) int {
	if p > float64(0) {
		return 1
	}
	return -1
}
