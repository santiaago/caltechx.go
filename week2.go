package main

import (
	"fmt"
	"math"
	"math/rand"
	//"sync"
	"time"
)

// r is a random variable used to generate random points through the program.
var r *rand.Rand

func sum(a []int) int {
	r := 0
	for _, v := range a {
		r += v
	}
	return r
}
func q1() {
	start := time.Now()
	nRuns := 100000
	nCoins := 1000
	nFlips := 10
	avgV1 := float64(0)
	avgVRand := float64(0)
	avgVMin := float64(0)
	//c1
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	//var wg sync.WaitGroup
	for run := 0; run < nRuns; run++ {
		//wg.Add(1)
		//go func() {
		coins := make([][]int, nCoins)
		for i, _ := range coins {
			coins[i] = make([]int, nFlips)
			for j := 0; j < len(coins[i]); j++ {
				coins[i][j] = r.Intn(2)
			}
		}
		v1 := float64(float64(sum(coins[0])) / float64(nFlips))
		randIndex := r.Intn(nCoins - 1)
		vRand := float64(float64(sum(coins[randIndex])) / float64(nFlips))
		vMin := math.Inf(1)
		for i, _ := range coins {
			vCurrent := float64(float64(sum(coins[i])) / float64(nFlips))
			if vCurrent < vMin {
				vMin = vCurrent
			}
		}
		avgV1 += v1
		avgVRand += vRand
		avgVMin += vMin
		//c <- 1
		//	wg.Done()
		//}()
	}
	// go func() {
	// 	wg.Wait()
	avgV1 = avgV1 / float64(nRuns)
	avgVRand = avgVRand / float64(nRuns)
	avgVMin = avgVMin / float64(nRuns)
	//	}()
	fmt.Printf("V1: %4.2f\nVRand: %4.2f\nVMin: %4.2f\n", avgV1, avgVRand, avgVMin)

	elapsed := time.Since(start)
	fmt.Printf("q1 took %s", elapsed)
}

func main() {
	fmt.Println("week 2")
	q1()
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
