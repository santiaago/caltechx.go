package main

import (
	"bufio"
	"fmt"
	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
	"log"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// measure will measure the time taken by function f to run and display it.
func measure(f func(), name string) {
	start := time.Now()
	f()
	elapsed := time.Since(start)
	fmt.Printf("%s took %4.2f seconds\n", name, elapsed.Seconds())
}

func getData(filename string) [][]float64 {

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var data [][]float64
	scanner := bufio.NewScanner(file)
	numberOfLines := 0
	for scanner.Scan() {

		split := strings.Split(scanner.Text(), " ")
		var line []string
		for _, s := range split {
			cell := strings.Replace(s, " ", "", -1)
			if len(cell) > 0 {
				line = append(line, cell)
			}
		}

		sample := make([]float64, 0)

		if x1, err := strconv.ParseFloat(line[0], 64); err != nil {
			fmt.Printf("x1 unable to parse line %d in file %s\n", numberOfLines, filename)
		} else {
			sample = append(sample, x1)
		}
		if x2, err := strconv.ParseFloat(line[1], 64); err != nil {
			fmt.Printf("x2 unable to parse line %d in file %s\n", numberOfLines, filename)
		} else {
			sample = append(sample, x2)
		}
		if y, err := strconv.ParseFloat(line[2], 64); err != nil {
			fmt.Printf("y unable to parse line %d in file %s\n", numberOfLines, filename)
		} else {
			sample = append(sample, y)
		}
		data = append(data, sample)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)

	}
	return data
}

type nonLinearTransformFunc func(x []float64) []float64

// non linear transformations

func phi0(x []float64) []float64 {
	return []float64{float64(x[0])}
}
func phi1(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1])}
}
func phi2(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2])}
}
func phi3(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2]), x[1] * x[1]}
}
func phi4(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2]), x[1] * x[1], x[2] * x[2]}
}
func phi5(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2]), x[1] * x[1], x[2] * x[2], x[1] * x[2]}
}
func phi6(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2]), x[1] * x[1], x[2] * x[2], x[1] * x[2], math.Abs(x[1] - x[2])}
}
func phi7(x []float64) []float64 {
	return []float64{float64(x[0]), float64(x[1]), float64(x[2]), x[1] * x[1], x[2] * x[2], x[1] * x[2], math.Abs(x[1] - x[2]), math.Abs(x[1] + x[2])}
}

func q1() {
	fns := []linreg.TransformFunc{phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7}

	ks := []int{3, 4, 5, 6, 7}

	data := getData("data/in.dta")
	for _, k := range ks {
		linreg := linreg.NewLinearRegression()

		linreg.InitializeFromData(data[:25])
		linreg.InitializeValidationFromData(data[25:])

		linreg.TransformFunction = fns[k]

		linreg.ApplyTransformation()
		linreg.ApplyTransformationOnValidation()
		linreg.Learn()
		eIn := linreg.Ein()
		eVal := linreg.EValIn()
		eOut, _ := linreg.EoutFromFile("data/out.dta")

		fmt.Printf("EVal = %f, for k = %d\n", eVal, k)
		fmt.Printf("EIn = %f, for k = %d\n", eIn, k)
		fmt.Printf("EOut = %f, for k = %d\n", eOut, k)
		fmt.Println()
	}
}

func q3() {
	fns := []linreg.TransformFunc{phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7}

	ks := []int{3, 4, 5, 6, 7}

	data := getData("data/in.dta")
	for _, k := range ks {
		linreg := linreg.NewLinearRegression()

		linreg.InitializeFromData(data[25:])
		linreg.InitializeValidationFromData(data[:25])

		linreg.TransformFunction = fns[k]

		linreg.ApplyTransformation()
		linreg.ApplyTransformationOnValidation()
		linreg.Learn()
		eIn := linreg.Ein()
		eVal := linreg.EValIn()
		eOut, _ := linreg.EoutFromFile("data/out.dta")

		fmt.Printf("EVal = %f, for k = %d\n", eVal, k)
		fmt.Printf("EIn = %f, for k = %d\n", eIn, k)
		fmt.Printf("EOut = %f, for k = %d\n", eOut, k)
		fmt.Println()
	}
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func q6() {
	runs := float64(10000)
	interval1 := linear.Interval{0, 1}
	interval2 := linear.Interval{0, 1}

	sumE1 := float64(0)
	sumE2 := float64(0)
	sumE := float64(0)
	for i := 0; i < int(runs); i++ {
		e1 := interval1.RandFloat()
		e2 := interval2.RandFloat()
		sumE1 += e1
		sumE2 += e2
		sumE += min(e1, e2)
	}
	fmt.Printf("e1 = %f, e2 = %f, e = %f\n", sumE1/runs, sumE2/runs, sumE/runs)
}
func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 7")
	fmt.Println("1")
	measure(q1, "q1")
	fmt.Println("2")
	fmt.Println("3")
	measure(q3, "q3")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	measure(q6, "q6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
