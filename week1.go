package main

import (
	"fmt"
	"github.com/santiaago/caltechx.go/pla"
)

// Take N = 10. How many iterations does it take on average for the PLA to
// converge for N = 10 training points?
func q7() {

	avgIterations := 0            // average of iterations that it takes for the PLA to converge
	avgDisagreement := float64(0) // average of disagreement between g (h function with Wn vector) and f
	runs := 1000                  // number of times we repeat the experiment
	pla := pla.NewPLA()

	for run := 0; run < runs; run++ {
		pla.Initialize()
		iterations := pla.Converge()
		avgDisagreement += pla.Disagreement()
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
	pla := pla.NewPLA()
	for run := 0; run < runs; run++ {
		pla.N = 100
		pla.Initialize()
		iterations := pla.Converge()
		avgDisagreement += pla.Disagreement()
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
