package main

import (
	"fmt"
	"github.com/santiaago/caltechx.go/biasAndVariance"
	"github.com/santiaago/caltechx.go/generalizationError"
	"runtime"
	"time"
)

// measure will measure the time taken by function f to run and display it.
func measure(f func(), name string) {
	start := time.Now()
	f()
	elapsed := time.Since(start)
	fmt.Printf("%s took %4.2f seconds\n", name, elapsed.Seconds())
}

func q1() {
	// generalization error:
	var genErr generalizationError.GeneralizationError
	genErr.Dvc = 10
	genErr.SetConfidence(0.95)
	genErr.Epsilon = 0.05

	fmt.Printf("The lower bound for a H with dvc = 10, confidence of 95 percent and epsilon of 0.05 is N = %v\n", genErr.LowerBound())
}

func q2() {
	// generalization error:
	var genErr generalizationError.GeneralizationError
	genErr.Dvc = 50
	genErr.SetDelta(0.05)
	genErr.Epsilon = 0.05
	fmt.Println("For dvc = 50, delta = 0.05, N = 10000")
	fmt.Printf("Original VC bound: \t\t\t%7.5f\n", genErr.VCBound(10000))
	fmt.Printf("Rademacher Penalty Bound: \t\t%7.5f\n", genErr.RademacherPenaltyBound(10000))
	fmt.Printf("Parrondo And Van Den Broek: \t\t%7.5f\n", genErr.ParrondoAndVanDenBroek(10000))
	fmt.Printf("Devroye: \t\t\t\t%7.5f\n", genErr.DevroyeLog(10000))
}

func q3() {
	// generalization error:
	var genErr generalizationError.GeneralizationError
	genErr.Dvc = 50
	genErr.SetDelta(0.05)
	genErr.Epsilon = 0.05
	fmt.Println("For dvc = 50, delta = 0.05, N = 5")
	fmt.Printf("Original VC bound: \t\t\t%7.5f\n", genErr.VCBound(5))
	fmt.Printf("Rademacher Penalty Bound: \t\t%7.5f\n", genErr.RademacherPenaltyBound(5))
	fmt.Printf("Parrondo And Van Den Broek: \t\t%7.5f\n", genErr.ParrondoAndVanDenBroek(5))
	fmt.Printf("Devroye: \t\t\t\t%7.5f\n", genErr.Devroye(5))
}

func q4() {
	bav := biasAndVariance.NewBiasAndVariance()
	bav.Learn()
	fmt.Printf("g(x) = %3.2f\n", bav.Slope)
	fmt.Printf("bias = %3.2f\n", bav.Bias)
	fmt.Printf("variance = %3.2f\n", bav.Variance)
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 4")
	measure(q1, "q1")
	fmt.Println("1 d")
	measure(q2, "q2")
	fmt.Println("2 d")
	measure(q3, "q3")
	fmt.Println("3 c")
	measure(q4, "q4")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
