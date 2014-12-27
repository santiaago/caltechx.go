package main

import (
	"fmt"
	"github.com/santiaago/caltechx.go/linreg"
	"math"
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

// non linear feature returns vector with form
//    (1, x1, x2, x1^2, x2^2, x1*x2, |x1 -x2|, |x1+ x2|)
func nonLinearFeature(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 8)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x1
	b[4] = x2 * x2
	b[5] = x1 * x2
	b[6] = math.Abs(x1 - x2)
	b[7] = math.Abs(x1 + x2)
	return b
}

func q2() {
	linreg := linreg.NewLinearRegression()
	linreg.InitializeFromFile("data/in.dta")
	linreg.TransformFunction = nonLinearFeature
	linreg.ApplyTransformation()
	linreg.Learn()
	ein := linreg.Ein()
	eout, _ := linreg.EoutFromFile("data/out.dta")
	fmt.Printf("Ein = %f, Eout = %f\n", ein, eout)
	linreg.K = -3
	linreg.LearnWeightDecay()
	eAugIn := linreg.EAugIn()
	eAugOut, _ := linreg.EAugOutFromFile("data/out.dta")
	fmt.Printf("Ein = %f, Eout = %f for k = %d\n", eAugIn, eAugOut, linreg.K)

	linreg.K = 3
	linreg.LearnWeightDecay()
	eAugIn = linreg.EAugIn()
	eAugOut, _ = linreg.EAugOutFromFile("data/out.dta")
	fmt.Printf("Ein = %f, Eout = %f for k = %d\n", eAugIn, eAugOut, linreg.K)

	k := []int{2, 1, 0, -1, -2}
	for _, ki := range k {
		linreg.K = ki
		linreg.LearnWeightDecay()
		eAugIn = linreg.EAugIn()
		eAugOut, _ = linreg.EAugOutFromFile("data/out.dta")
		fmt.Printf("Ein = %f, Eout = %f for k = %d\n", eAugIn, eAugOut, ki)
	}

	minK := 0
	minEAug := float64(1000)
	for i := -10; i < 10; i++ {
		linreg.K = i
		linreg.LearnWeightDecay()
		eAugOut, _ = linreg.EAugOutFromFile("data/out.dta")
		if minEAug > eAugOut {
			minK = i
			minEAug = eAugOut
		}
	}
	fmt.Printf("mininum Eout = %f with k = %d for k in {-10; 10}\n", minEAug, minK)
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 6")
	fmt.Println("1")
	fmt.Println("2")
	measure(q2, "q2")
	fmt.Println("3")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
