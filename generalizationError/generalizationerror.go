package generalizationError

import (
	"math"
)

// There are various bounds for generalization error.
// This structure holds 4 functions for calculating these bounds:
//  - Original VC bound
//  - Rademacher Penalty Bound
//  - Parrondo and Van den Broek
//  - Devroye
type GeneralizationError struct {
	Dvc        int     // VC Dimention.
	Delta      float64 // Upper bound on the probability that generalization error will be more than a specified value.
	Confidence float64 // 1 - delta = Confidence that generalization error will be at most a specified value.
	Epsilon    float64 // Generalization error tolerance.
	Margin     float64 // Margin of error.
}

// M_H is the growth function: counts the most dicotomies on any N points following:
// M_H(N) <= 2**N if N <= VC dimention
// M_H(N) <= N**Dvc if N > VC dimention
func (g *GeneralizationError) M_H(n int) float64 {
	if n <= g.Dvc {
		return math.Pow(float64(2), float64(n))
	}
	return math.Pow(float64(n), float64(g.Dvc))
}

// M_HLog is the growth function: counts the most dicotomies on any N points following:
// M_HLog(N) <= N*Ln(2) if N <= VC dimention
// M_HLog(N) <= Dvc*Ln(N) if N > VC dimention
func (g *GeneralizationError) M_HLog(n int) float64 {
	if n <= g.Dvc {
		return float64(n) * math.Log(2)
	}
	return float64(g.Dvc) * math.Log(float64(n))
}

// VCBound is the original Vapnik Chervonenkis bound
// epsilon ≤ sqrt(8/N ln 4mH(2N))
func (g *GeneralizationError) VCBound(n int) float64 {
	return math.Sqrt(float64(8) * float64(float64(1)/float64(n)) * math.Log(float64(4)*g.M_H(2*n)/g.Delta))
}

// RademacherPenaltyBound:
// epsilon ≤ sqrt(2 ln(2NmH(N))/N) + sqrt(2/N ln 1/δ +) + 1/N
func (g *GeneralizationError) RademacherPenaltyBound(n int) float64 {
	return math.Sqrt(float64(2)*math.Log(float64(2)*float64(n)*g.M_H(n))/float64(n)) + math.Sqrt(float64(2)/float64(n)*math.Log(float64(1)/float64(g.Delta))) + float64(1)/float64(n)
}

// ParrondoAndVanDenBroek:
// epsilon ≤ sqrt(1/N(2epsilon + ln 6mH(2N)/δ))
func (g *GeneralizationError) ParrondoAndVanDenBroek(n int) float64 {
	f := func(n int, eps float64) float64 {
		return math.Sqrt((float64(1) / float64(n)) * (float64(2)*eps + math.Log(float64(6)*g.M_H(2*n)/g.Delta)))
	}
	return f(n, 1)
	e1 := f(n, 0)
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		// if e1-e0 <= margin {
		// 	break
		// }
	}
	return e1
}

// DevroyeLog:
// epsilon ≤ sqrt(1/2N (4epsilon(1 + epsilon) + ln 4mH(N^2)/δ)
func (g *GeneralizationError) DevroyeLog(n int) float64 {
	f := func(n int, eps float64) float64 {
		return math.Sqrt((float64(1) / float64(2*n)) * (float64(4)*eps*float64(1+eps) + math.Log(float64(4)) + g.M_HLog(n*n) - math.Log(g.Delta)))
	}
	return f(n, 1)
	e1 := f(n, 0)
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		// if e1-e0 <= margin {
		// 	break
		// }
	}
	return e1
}

// Devroye:
// epsilon ≤ sqrt(1/2N (4epsilon(1 + epsilon) + ln 4mH(N^2)/δ)
func (g *GeneralizationError) Devroye(n int) float64 {
	f := func(n int, eps float64) float64 {
		a := (float64(1) / float64(2*n))
		return math.Sqrt(a * (float64(4)*eps*float64(1+eps) + (math.Log(float64(4) * g.M_H(n*n) / g.Delta))))
	}
	return f(n, 1)
	e1 := f(n, 0)
	for i := 0; i < 10000; i++ {
		e0 := e1
		e1 = f(n, e0)
		// if e1-e0 <= margin {
		// 	break
		// }
	}
	return e1
}

// SetConfidence will set the confidence variable and it's oposite Delta as 1 - Confidence.
func (g *GeneralizationError) SetConfidence(c float64) {
	g.Confidence = c
	g.Delta = 1 - c
}

// SetConfidence will set the delta variable and it's oposite confidence as 1 - delta.
func (g *GeneralizationError) SetDelta(d float64) {
	g.Delta = d
	g.Confidence = 1 - d
}

// LowerBound computes the minimum sample size necessary to satisfy the VC inequality
// N >= (8/epsilon^2)ln((4*M_H(2*N))/delta)
// It uses an iterative method for solving N = f(N).
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
