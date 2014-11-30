package main

import (
	"fmt"
	"github.com/santiaago/caltechx.go/hoeffding"
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
	hoef := hoeffding.NewHoeffdingExperiment()
	hoef.M = 1
	Ns := [...]int{500, 1000, 1500, 2000}
	for i := range Ns {
		hoef.N = Ns[i]
		fmt.Printf("Hoeffding inequality for N = %v is %7.5f\n", Ns[i], hoef.Compute())
	}
}

func q2() {
	hoef := hoeffding.NewHoeffdingExperiment()
	hoef.M = 10
	Ns := [...]int{500, 1000, 1500, 2000}
	for i := range Ns {
		hoef.N = Ns[i]
		fmt.Printf("Hoeffding inequality for M = %v, N = %v is %7.5f\n", hoef.M, hoef.N, hoef.Compute())
	}
}

func q3() {
	hoef := hoeffding.NewHoeffdingExperiment()
	hoef.M = 100
	Ns := [...]int{500, 1000, 1500, 2000}
	for i := range Ns {
		hoef.N = Ns[i]
		fmt.Printf("Hoeffding inequality for M = %v, N = %v is %7.5f\n", hoef.M, hoef.N, hoef.Compute())
	}
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 3")
	measure(q1, "q1")
	fmt.Println("1")
	measure(q2, "q2")
	fmt.Println("2")
	measure(q3, "q3")
	fmt.Println("3")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
