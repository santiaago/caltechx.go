package main

import (
	"bufio"
	"fmt"
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

// non linear transformation
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
	fns := []linreg.TransformFunc{
		phi0,
		phi1,
		phi2,
		phi3,
		phi4,
		phi5,
		phi6,
		phi7,
	}

	ks := []int{3, 4, 5, 6, 7}

	data := getData("data/in.dta")
	for _, k := range ks {
		fmt.Println(k)
		linreg := linreg.NewLinearRegression()

		linreg.InitializeFromData(data[:25])
		linreg.InitializeValidationFromData(data[25:])

		linreg.TransformFunction = fns[k]

		linreg.ApplyTransformation()
		linreg.ApplyTransformationOnValidation()
		linreg.Learn()
		eVal := linreg.EValIn()
		fmt.Printf("EVal = %f, for k = %d\n", eVal, k)
	}
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 7")
	fmt.Println("1")
	measure(q1, "q1")
	fmt.Println("2")
	fmt.Println("3")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
