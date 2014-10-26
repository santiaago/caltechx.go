package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
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

func q1() {
	start := time.Now()
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

	elapsed := time.Since(start)
	fmt.Printf("q1 took %4.2f seconds\n", elapsed.Seconds())
}

func q1ll() {

	start := time.Now()

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

	elapsed := time.Since(start)
	fmt.Printf("q1 parallel took %4.2f seconds\n", elapsed.Seconds())
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 2")
	//q1()
	//q1ll()
	fmt.Println("1 b")
	fmt.Println("2 d")
	fmt.Println("3 ")
	fmt.Println("4 ")
	fmt.Println("5 ")
	fmt.Println("6 ")
	fmt.Println("7 ")
	fmt.Println("8 ")
	fmt.Println("9 ")
	fmt.Println("10 ")
}
