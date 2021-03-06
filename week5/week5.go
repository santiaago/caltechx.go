package main

import (
	"fmt"
	GD "github.com/santiaago/caltechx.go/gradientDescent"
	"github.com/santiaago/caltechx.go/linreg"
	"github.com/santiaago/caltechx.go/logreg"
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
	ns := [...]int{10, 25, 100, 500, 1000}
	for i := range ns {
		fmt.Println(ns[i], " : ", linreg.LinearRegressionError(ns[i], float64(0.1), 8) > float64(0.008))
	}
}

func q5() {
	var gd GD.GradientDescent
	gd.U = float64(1)
	gd.V = float64(1)
	gd.Eta = float64(0.1)
	gd.ErrorLimit = 10e-14
	fmt.Println("iterations needed to fall under the limit, ", gd.E())
	fmt.Println("error: ", gd.Err())
	fmt.Println("U:", gd.U, " V:", gd.V)
}

func q7() {
	var cd GD.CoordinateDescent
	cd.U = float64(1)
	cd.V = float64(1)
	cd.Eta = float64(0.1)
	cd.IterationLimit = 15
	fmt.Println("iterations needed to fall under the limit, ", cd.E())
	fmt.Println("error: ", cd.Err())
	fmt.Println("U:", cd.U, " V:", cd.V)
}

func q8() {
	eout := float64(0)
	epochs := 0
	for i := 0; i < 100; i++ {
		lg := logreg.NewLogisticRegression()
		lg.Initialize()
		lg.Learn()
		eout += lg.Eout()
		epochs += lg.Epochs
	}
	fmt.Println("logistic regression: eout: ", eout/float64(100))
	fmt.Println("logistic regression: number of epochs: ", float64(epochs)/float64(100))
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 5")
	measure(q1, "q1")
	fmt.Println("1")
	fmt.Println("2")
	fmt.Println("3")
	fmt.Println("4")
	fmt.Println("5")
	measure(q5, "q5")
	fmt.Println("6")
	measure(q7, "q7")
	fmt.Println("7")
	measure(q8, "q8")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
