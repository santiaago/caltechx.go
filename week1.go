package main

import (
	"fmt"
	"math/rand"
	"time"
)

// type point struct {
// 	x0 float64 // artificial coordinate for the perceptron learning algorithm
// 	x  float64
// 	y  float64
// }

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

type point [3]float64

func (pt *point) print() {
	p := *pt
	fmt.Printf("point: x0:%v \tx1:%v \tx2:%v\n", p[0], p[1], p[2])
}

func evaluate(f linear_func, p point) int {
	if p[2] < f(p[1]) {
		return -1
	}
	return 1
}

func sign(p float64) int {
	if p > float64(0) {
		return 1
	}
	return -1
}

// perceptron implements h(x) = sign(w'x)
func h(x point, w point) int {
	if len(x) != len(w) {
		fmt.Println("Panic: vectors x and w should be of same size.")
		panic(x)
	}
	var res float64 = 0
	for i := 0; i < len(w); i++ {
		res = res + w[i]*x[i]
	}
	return sign(res)
}

// build misclassified set.
func build_misclassified_set(Xn []point, Yn []int, Wn []point) []int {
	var set []int
	for i := 0; i < len(Wn); i++ {
		if h(Xn[i], Wn[i]) != Yn[i] {
			set = append(set, i) //set.Append(i)
		}
	}
	return set
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
	//var Xn [10]point //[3]float64
	Xn := make([]point, 10)
	Yn := make([]int, 10)
	Wn := make([]point, 10)
	for i := 0; i < 10; i++ {
		//Wn[i] = point{0, 0, 0} no need to do this as it is automatically set to zero.
		Xn[i][0] = float64(1)
		Xn[i][1] = randInInterval(-1, 1)
		Xn[i][2] = randInInterval(-1, 1)
		Xn[i].print()
		Yn[i] = evaluate(f.getFunc(), Xn[i])
	}

	for i := 0; i < 10; i++ {
		fmt.Printf("Y%v : %v\n", i, Yn[i])
	}

	n_iterations := 0
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for {
		fmt.Printf("iteration #%v\n", n_iterations)
		// pick a misclasified number
		// first build a misclassified set:
		misclasified_set := build_misclassified_set(Xn, Yn, Wn)
		fmt.Println("number of misclassified points: ", len(misclasified_set))
		for i := 0; i < len(Wn); i++ {
			fmt.Printf("W0: %v \t W1: %v \t W2: %v\n", Wn[0], Wn[1], Wn[2])
		}
		if len(misclasified_set) == 0 {
			break
		}
		rand_point := r.Intn(len(misclasified_set))
		// update the weight vector:
		// w <-- w + YnXn
		Wn[rand_point][0] = Wn[rand_point][0] + float64(Yn[rand_point])*Xn[rand_point][0]
		Wn[rand_point][1] = Wn[rand_point][1] + float64(Yn[rand_point])*Xn[rand_point][1]
		Wn[rand_point][2] = Wn[rand_point][2] + float64(Yn[rand_point])*Xn[rand_point][2]

		n_iterations += 1
		if n_iterations > 10 {
			break
		}
	}
}
