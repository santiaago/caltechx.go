package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/santiaago/caltechx.go/linreg"
	"github.com/santiaago/caltechx.go/pla"
	"github.com/santiaago/ml"
)

func sum(a []int) int {
	r := 0
	for _, v := range a {
		r += v
	}
	return r
}

type experiment struct {
	NRuns    int
	NCoins   int
	NFlips   int
	AvgV1    float64
	AvgVRand float64
	AvgVMin  float64
}

// measure will measure the time taken by function f to run and display it.
func measure(f func(), name string) {
	start := time.Now()
	f()
	elapsed := time.Since(start)
	fmt.Printf("%s took %4.2f seconds\n", name, elapsed.Seconds())
}

func q1() {
	e := experiment{NRuns: 100000, NCoins: 1000, NFlips: 10}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	var sumV1, sumVRand, sumVMin float64
	for run := 0; run < e.NRuns; run++ {
		coins := make([][]int, e.NCoins)
		for i, _ := range coins {
			coins[i] = make([]int, e.NFlips)
			for j := 0; j < len(coins[i]); j++ {
				coins[i][j] = r.Intn(2)
			}
		}
		v1 := float64(float64(sum(coins[0])) / float64(e.NFlips))
		randIndex := r.Intn(e.NCoins - 1)
		vRand := float64(float64(sum(coins[randIndex])) / float64(e.NFlips))
		vMin := math.Inf(1)
		for i, _ := range coins {
			vCurrent := float64(float64(sum(coins[i])) / float64(e.NFlips))
			if vCurrent < vMin {
				vMin = vCurrent
			}
		}
		sumV1 += v1
		sumVRand += vRand
		sumVMin += vMin
	}
	e.AvgV1 = sumV1 / float64(e.NRuns)
	e.AvgVRand = sumVRand / float64(e.NRuns)
	e.AvgVMin = sumVMin / float64(e.NRuns)
	fmt.Printf("V1: %4.2f\nVRand: %4.2f\nVMin: %4.2f\n", e.AvgV1, e.AvgVRand, e.AvgVMin)
}

func q1cc() {
	e := experiment{NRuns: 100000, NCoins: 1000, NFlips: 10}

	var wg sync.WaitGroup

	chanV1 := make(chan float64, e.NRuns)
	chanVRand := make(chan float64, e.NRuns)
	chanVMin := make(chan float64, e.NRuns)

	for run := 0; run < e.NRuns; run++ {
		wg.Add(1)
		go func() {
			r := rand.New(rand.NewSource(time.Now().UnixNano()))
			coins := make([][]int, e.NCoins)
			for i, _ := range coins {
				coins[i] = make([]int, e.NFlips)
				for j := 0; j < len(coins[i]); j++ {
					coins[i][j] = r.Intn(2)
				}
			}
			v1 := float64(float64(sum(coins[0])) / float64(e.NFlips))
			randIndex := r.Intn(e.NCoins - 1)
			vRand := float64(float64(sum(coins[randIndex])) / float64(e.NFlips))
			vMin := math.Inf(1)
			for i, _ := range coins {
				vCurrent := float64(float64(sum(coins[i])) / float64(e.NFlips))
				if vCurrent < vMin {
					vMin = vCurrent
				}
			}
			chanV1 <- v1
			chanVRand <- vRand
			chanVMin <- vMin
			wg.Done()
		}()
	}

	wg.Wait()
	var sumV1, sumVRand, sumVMin float64
	for i := 0; i < e.NRuns; i++ {
		sumV1 += <-chanV1
	}
	for i := 0; i < e.NRuns; i++ {
		sumVRand += <-chanVRand
	}
	for i := 0; i < e.NRuns; i++ {
		sumVMin += <-chanVMin
	}

	e.AvgV1 = sumV1 / float64(e.NRuns)
	e.AvgVRand = sumVRand / float64(e.NRuns)
	e.AvgVMin = sumVMin / float64(e.NRuns)
	fmt.Printf("V1: %4.2f\nVRand: %4.2f\nVMin: %4.2f\n", e.AvgV1, e.AvgVRand, e.AvgVMin)
}

func q5() {
	runs := 1000 // number of times we repeat the experiment
	linreg := linreg.NewLinearRegression()
	linreg.N = 100
	var avgEin, avgEout float64

	for run := 0; run < runs; run++ {
		linreg.Initialize()

		linreg.Learn()
		avgEin += linreg.Ein()
		avgEout += linreg.Eout()
	}
	avgEin = avgEin / float64(runs)
	avgEout = avgEout / float64(runs)
	fmt.Printf("average of In sample error 'Ein' for Linear regresion for N = 100 is %4.2f\n", avgEin)
	fmt.Printf("average of Out of sample error 'Eout' for Linear regresion for N = 100 is %4.2f\n", avgEout)
}

func q7() {
	runs := 1000 // number of times we repeat the experiment
	linreg := linreg.NewLinearRegression()
	linreg.N = 10
	iterations := 0
	for run := 0; run < runs; run++ {
		linreg.Initialize()

		linreg.Learn()
		pla := pla.NewPLA()
		pla.Initialize()
		for i := 0; i < len(pla.Wn); i++ {
			pla.Wn[i] = linreg.Wn[i]
		}
		iterations += pla.Converge()
	}
	avgIterations := float64(iterations) / float64(runs)
	fmt.Printf("average for PLA to converge for N = %v with initial weight computed by linear regression is %4.2f\n", linreg.N, avgIterations)
}

// non linear transformation
func f(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(x1*x1 + x2*x2 - 0.6)
}

func q8() {

	runs := 1000 // number of times we repeat the experiment
	linreg := linreg.NewLinearRegression()
	linreg.N = 1000
	linreg.RandomTargetFunction = false
	linreg.TwoParams = true
	linreg.Noise = 0.1
	var avgEin float64

	for run := 0; run < runs; run++ {

		linreg.TargetFunction = f //non linear function
		linreg.Initialize()
		linreg.Learn()
		avgEin += linreg.Ein()
	}

	avgEin = avgEin / float64(runs)
	fmt.Printf("average of In sample error 'Ein' for Linear regresion for N = 100 is %4.2f\n", avgEin)
}

// non linear feature returns vector with form (1, x1, x2, x1x2, x1^2, x2^2)
func nonLinearFeature(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = a[1]
	b[2] = a[2]
	b[3] = a[1] * a[2]
	b[4] = a[1] * a[1]
	b[5] = a[2] * a[2]
	return b
}

func a(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1*x1 + 1.5*x2*x2)
}

func b(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1*x1 + 15*x2*x2)
}

func c(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 15*x1*x1 + 1.5*x2*x2)
}

func d(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(-1 - 1.5*x1 + 0.08*x2 + 0.13*x1*x2 + 0.05*x1*x1 + 0.05*x2*x2)
}

func e(x ...float64) float64 {
	x1 := x[0]
	x2 := x[1]
	return ml.Sign(-1 - 0.05*x1 + 0.08*x2 + 1.5*x1*x2 + 0.15*x1*x1 + 0.15*x2*x2)
}

func q9() {

	runs := 1000 // number of times we repeat the experiment
	linreg := linreg.NewLinearRegression()
	linreg.N = 1000
	linreg.RandomTargetFunction = false
	linreg.TwoParams = true
	linreg.Noise = 0.1
	var aAvgIn, bAvgIn, cAvgIn, dAvgIn, eAvgIn float64
	var aAvgOut, bAvgOut, cAvgOut, dAvgOut, eAvgOut float64

	var AvgEout float64
	for run := 0; run < runs; run++ {

		linreg.TargetFunction = f //non linear function
		linreg.Initialize()
		linreg.TransformDataSet(nonLinearFeature, 6)
		linreg.Learn()

		aAvgIn += linreg.CompareInSample(a, 2)
		bAvgIn += linreg.CompareInSample(b, 2)
		cAvgIn += linreg.CompareInSample(c, 2)
		dAvgIn += linreg.CompareInSample(d, 2)
		eAvgIn += linreg.CompareInSample(e, 2)

		aAvgOut += linreg.CompareOutOfSample(a, 2)
		bAvgOut += linreg.CompareOutOfSample(b, 2)
		cAvgOut += linreg.CompareOutOfSample(c, 2)
		dAvgOut += linreg.CompareOutOfSample(d, 2)
		eAvgOut += linreg.CompareOutOfSample(e, 2)

		AvgEout += linreg.Eout()
	}

	aAvgIn = aAvgIn / float64(runs)
	bAvgIn = bAvgIn / float64(runs)
	cAvgIn = cAvgIn / float64(runs)
	dAvgIn = dAvgIn / float64(runs)
	eAvgIn = eAvgIn / float64(runs)

	aAvgOut = aAvgOut / float64(runs)
	bAvgOut = bAvgOut / float64(runs)
	cAvgOut = cAvgOut / float64(runs)
	dAvgOut = dAvgOut / float64(runs)
	eAvgOut = eAvgOut / float64(runs)

	AvgEout = AvgEout / float64(runs)

	fmt.Printf("average of difference in sample between a() and the linear regression hypothesis is %5.3f\n", aAvgIn)
	fmt.Printf("average of difference in sample between b() and the linear regression hypothesis is %5.3f\n", bAvgIn)
	fmt.Printf("average of difference in sample between c() and the linear regression hypothesis is %5.3f\n", cAvgIn)
	fmt.Printf("average of difference in sample between d() and the linear regression hypothesis is %5.3f\n", dAvgIn)
	fmt.Printf("average of difference in sample between e() and the linear regression hypothesis is %5.3f\n", eAvgIn)

	fmt.Printf("average of difference out of sample between a() and the linear regression hypothesis is %7.5f\n", aAvgOut)
	fmt.Printf("average of difference out of sample between b() and the linear regression hypothesis is %7.5f\n", bAvgOut)
	fmt.Printf("average of difference out of sampleb etween c() and the linear regression hypothesis is %7.5f\n", cAvgOut)
	fmt.Printf("average of difference out of sample between d() and the linear regression hypothesis is %7.5f\n", dAvgOut)
	fmt.Printf("average of difference out of sample between e() and the linear regression hypothesis is %7.5f\n", eAvgOut)

	fmt.Printf("average Eout of the hypothesis learn through linear regression with noise of %4.2f is %4.3f\n", linreg.Noise, AvgEout)
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 2")
	//measure(q1, "q1")
	//measure(q1cc, "q1 concurrent")
	fmt.Println("1")
	fmt.Println("2")
	fmt.Println("3")
	fmt.Println("4")
	//measure(q5, "q5")
	fmt.Println("5")
	fmt.Println("6")
	//measure(q7, "q7")
	fmt.Println("7")
	//measure(q8, "q8")
	fmt.Println("8")
	measure(q9, "q9")
	fmt.Println("9")
	fmt.Println("10")
}
