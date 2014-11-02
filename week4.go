package main

import (
	"fmt"
	"math"
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

type BoundFunction func(N int) int

type GeneralizationError struct {
	Dvc           int           // VC Dimentions.
	Delta         float64       // Upper bound on the probability that generalization error will be more than a specified value.
	Confidence    float64       // 1 - delta = Confidence that generalization error will be at most a specified value.
	Epsilon       float64       // Generalization error tolerance.
	BoundFunction BoundFunction // bound function
	Margin        float64       // margin of error
}

func (g *GeneralizationError) M_H(n int) float64 {
	if n <= g.Dvc {
		return math.Pow(float64(2), float64(n))
	}
	return math.Pow(float64(n), float64(g.Dvc))
}

func (g *GeneralizationError) M_HLog(n int) float64 {
	if n <= g.Dvc {
		return float64(n) * math.Log(2)
	}
	return float64(g.Dvc) * math.Log(float64(n))
}

func (g *GeneralizationError) VCBound(n int) float64 {
	return math.Sqrt(float64(8) * float64(float64(1)/float64(n)) * math.Log(float64(4)*g.M_H(2*n)/g.Delta))
}

func (g *GeneralizationError) RademacherPenaltyBound(n int) float64 {
	return math.Sqrt(float64(2)*math.Log(float64(2)*float64(n)*g.M_H(n))/float64(n)) + math.Sqrt(float64(2)/float64(n)*math.Log(float64(1)/float64(g.Delta))) + float64(1)/float64(n)
}

func (g *GeneralizationError) ParrondoAndVanDenBroek(n int) float64 {
	f := func(n int, eps float64) float64 {
		return math.Sqrt((float64(1) / float64(n)) * (float64(2)*eps + math.Log(float64(6)*g.M_H(2*n)/g.Delta)))
	}
	return f(n, 1)
	e1 := f(n, 0)
	margin := 0.0001
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		if e1-e0 <= margin {
			break
		}
	}
	return e1
}

func (g *GeneralizationError) DevroyeLog(n int) float64 {
	f := func(n int, eps float64) float64 {
		return math.Sqrt((float64(1) / float64(2*n)) * (float64(4)*eps*float64(1+eps) + math.Log(float64(4)) + g.M_HLog(n*n) - math.Log(g.Delta)))
	}
	return f(n, 1)
	e1 := f(n, 0)
	margin := 0.0001
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		if e1-e0 <= margin {
			break
		}
	}
	return e1
}

func (g *GeneralizationError) Devroye(n int) float64 {
	f := func(n int, eps float64) float64 {
		a := (float64(1) / float64(2*n))
		return math.Sqrt(a * (float64(4)*eps*float64(1+eps) + (math.Log(float64(4) * g.M_H(n*n) / g.Delta))))
	}
	return f(n, 1)
	e1 := f(n, 0)
	margin := 0.0001
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		if e1-e0 <= margin {
			break
		}

	}
	return e1
}

func (g *GeneralizationError) SetConfidence(c float64) {
	g.Confidence = c
	g.Delta = 1 - c
}

func (g *GeneralizationError) SetDelta(d float64) {
	g.Delta = d
	g.Confidence = 1 - d
}

func (g *GeneralizationError) LowerBound() int {
	n0 := 1000
	f := func(n int) int {
		x := (float64(4) * (math.Pow(float64(2*n), float64(g.Dvc)) + float64(1))) / g.Delta
		res := float64(8) * float64(1) / (g.Epsilon * g.Epsilon) * math.Log(x)
		return int(res)
	}

	n1 := f(n0)
	for i := 0; i < 10000; i++ {
		n0 = n1
		n1 = f(n0)
	}
	return n1
}

func q1() {
	// generalization error:
	var genErr GeneralizationError
	genErr.Dvc = 10
	genErr.SetConfidence(0.95)
	genErr.Epsilon = 0.05

	fmt.Printf("The lower bound for a H with dvc = 10, confidence of 95 percent and epsilon of 0.05 is N = %v\n", genErr.LowerBound())
}

func q2() {
	// generalization error:
	var genErr GeneralizationError
	genErr.Dvc = 50
	genErr.SetDelta(0.05)
	genErr.Epsilon = 0.05
	fmt.Println("For dvc = 50, delta = 0.05, N = 10000")
	fmt.Printf("Original VC bound: \t\t\t%7.5f\n", genErr.VCBound(10000))
	fmt.Printf("Rademacher Penalty Bound: \t\t%7.5f\n", genErr.RademacherPenaltyBound(10000))
	fmt.Printf("Parrondo And Van Den Broek: \t\t%7.5f\n", genErr.ParrondoAndVanDenBroek(10000))
	fmt.Printf("Devroye: \t\t\t\t%7.5f\n", genErr.DevroyeLog(10000))
}

func q3() {
	// generalization error:
	var genErr GeneralizationError
	genErr.Dvc = 50
	genErr.SetDelta(0.05)
	genErr.Epsilon = 0.05
	fmt.Println("For dvc = 50, delta = 0.05, N = 5")
	fmt.Printf("Original VC bound: \t\t\t%7.5f\n", genErr.VCBound(5))
	fmt.Printf("Rademacher Penalty Bound: \t\t%7.5f\n", genErr.RademacherPenaltyBound(5))
	fmt.Printf("Parrondo And Van Den Broek: \t\t%7.5f\n", genErr.ParrondoAndVanDenBroek(5))
	fmt.Printf("Devroye: \t\t\t\t%7.5f\n", genErr.DevroyeLog(5))
}

func main() {
	fmt.Println("Num CPU: ", runtime.NumCPU())
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("week 4")
	measure(q1, "q1")
	fmt.Println("1 d")
	measure(q2, "q2")
	fmt.Println("2 d")
	measure(q3, "q3")
	fmt.Println("3 c")
	fmt.Println("4")
	fmt.Println("5")
	fmt.Println("6")
	fmt.Println("7")
	fmt.Println("8")
	fmt.Println("9")
	fmt.Println("10")
}
