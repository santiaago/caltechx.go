package main

import (
	"fmt"
	"github.com/santiaago/caltechx.go/linreg"
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

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 5")
	measure(q1, "q1")
	fmt.Println("1 c")
	fmt.Println("2 d")
	fmt.Println("3 c")
	fmt.Println("4 e")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
