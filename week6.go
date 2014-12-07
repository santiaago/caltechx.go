package main

import (
	"fmt"
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

func q2() {
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
	measure(q5, "q5")
	fmt.Println("6")
	measure(q7, "q7")
	fmt.Println("7")
	measure(q8, "q8")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
