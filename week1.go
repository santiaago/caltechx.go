package main

import (
	"fmt"
	"math/rand"
	"time"
)

type point struct {
	x0 float64 // artificial coordinate for the perceptron learning algorithm
	x  float64
	y  float64
}

type linear_function struct {
	a float64
	b float64
}

type linear_func func(x float64) float64

// computes a random number in a given interval
func randInInterval(min int, max int) float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := float64(max - min)
	return r.Float64()*size + float64(min)
}

func randLine() linear_function {
	a := randInInterval(-1, 1)
	b := randInInterval(-1, 1)
	return linear_function{a, b}
}

func (l *linear_function) getFunc() linear_func {
	return func(x float64) float64 {
		return x*l.a + l.b
	}
}

func (l *linear_function) print() {
	fmt.Printf("func: %vX + %v\n", l.a, l.b)
}

func (p *point) print() {
	fmt.Printf("point: x:%v \ty:%v\n", p.x, p.y)
}

func evaluate(f linear_func, p point) int {
	if p.y < f(p.x) {
		return -1
	}
	return 1
}

func main() {
	fmt.Println("week 1")
	fmt.Println("1 d")
	fmt.Println("2 a")
	fmt.Println("3 e")
	fmt.Println("4 b")
	fmt.Println("5 c")
	fmt.Println("6 e")
	// 7: perceptron learning algorithm
	fmt.Println(randInInterval(-1, 1))
	f := randLine()
	f.print()
	var Xn [10]point
	var Yn [10]int
	var Wn [10]point
	for i := 0; i < 10; i++ {
		Wn[i] = point{0, 0, 0}
		Xn[i] = point{1, randInInterval(-1, 1), randInInterval(-1, 1)}
		Xn[i].print()
		Yn[i] = evaluate(f.getFunc(), Xn[i])
	}

	for i := 0; i < 10; i++ {
		fmt.Printf("Y%v : %v\n", i, Yn[i])
	}

}
