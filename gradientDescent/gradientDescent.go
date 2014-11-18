package GradientDescent

import (
	"math"
)

type GradientDescent struct {
	Eta        float64 // learning rate
	U          float64
	V          float64
	ErrorLimit float64
}

type CoordinateDescent struct {
	Eta            float64 // learning rate
	U              float64
	V              float64
	ErrorLimit     float64
	IterationLimit int
}

func (cd *CoordinateDescent) E() int {
	iteration := 0
	for {
		if cd.IterationLimit < iteration {
			break
		}

		cd.U = cd.U - cd.Eta*cd.DerivU()
		dV := cd.V - cd.Eta*cd.DerivV()
		cd.V = dV
		iteration++
	}
	return iteration
}

func (gd *GradientDescent) E() int {
	iteration := 0

	for {
		if gd.Err() < gd.ErrorLimit {
			break
		}
		dU := gd.U - gd.Eta*gd.DerivU()
		dV := gd.V - gd.Eta*gd.DerivV()
		gd.U = dU
		gd.V = dV

		iteration++
	}
	return iteration
}

func (gd *GradientDescent) DerivU() float64 {
	return float64(2) * (math.Exp(gd.V) + float64(2)*gd.V*math.Exp(-gd.U)) * (gd.U*math.Exp(gd.V) - float64(2)*gd.V*math.Exp(-gd.U))
}

func (gd *GradientDescent) DerivV() float64 {
	return float64(2) * (gd.U*math.Exp(gd.V) - float64(2)*math.Exp(-gd.U)) * (gd.U*math.Exp(gd.V) - float64(2)*gd.V*math.Exp(-gd.U))
}

func (cd *CoordinateDescent) DerivU() float64 {
	return float64(2) * (math.Exp(cd.V) + float64(2)*cd.V*math.Exp(-cd.U)) * (cd.U*math.Exp(cd.V) - float64(2)*cd.V*math.Exp(-cd.U))
}

func (cd *CoordinateDescent) DerivV() float64 {
	return float64(2) * (cd.U*math.Exp(cd.V) - float64(2)*math.Exp(-cd.U)) * (cd.U*math.Exp(cd.V) - float64(2)*cd.V*math.Exp(-cd.U))
}

// Non linear error surface:
// (u^v - 2v^-u)^2
func (gd *GradientDescent) Err() float64 {
	return math.Pow(gd.U*math.Exp(gd.V)-float64(2)*gd.V*math.Exp(-gd.U), 2)
}

func (cd *CoordinateDescent) Err() float64 {
	return math.Pow(cd.U*math.Exp(cd.V)-float64(2)*cd.V*math.Exp(-cd.U), 2)
}
