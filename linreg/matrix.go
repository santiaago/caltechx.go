package linreg

import (
	"fmt"
	"math"
)

type matrix [][]float64

func (pm *matrix) inverse() matrix {
	m := *pm
	n := len(m)
	if n != len(m[0]) {
		fmt.Println("Panic: matrix should be square")
		panic(m)
	}
	x := make([][]float64, n) // inverse matrix to return
	for i := 0; i < n; i++ {
		x[i] = make([]float64, n)
	}
	LU, p := LUPDecomposition(m)

	// Solve AX = e for each column ei of the identity matrix using LUP decomposition
	for i := 0; i < n; i++ {
		e := make([]float64, n)
		e[i] = 1
		solve := LUPSolve(LU, p, e)
		for j := 0; j < len(solve); j++ {
			x[j][i] = solve[j]
		}
	}
	return x
}

func LUPSolve(LU matrix, pi []int, b []float64) []float64 {
	n := len(LU)
	x := make([]float64, n)
	y := make([]float64, n)
	var suml, sumu, lij float64

	// solve for y using formward substitution
	for i := 0; i < n; i++ {
		suml = float64(0)
		for j := 0; j <= i-1; j++ {
			if i == j {
				lij = 1
			} else {
				lij = LU[i][j]
			}
			suml = suml + (lij * y[j])
		}
		y[i] = b[pi[i]] - suml
	}
	//Solve for x by using back substitution
	for i := n - 1; i >= 0; i-- {
		sumu = 0
		for j := i + 1; j < n; j++ {
			sumu = sumu + (LU[i][j] * x[j])
		}
		x[i] = (y[i] - sumu) / LU[i][i]
	}
	return x
}

// Perform LUP decomposition on a matrix A.
// Return L and U as a single matrix(double[][]) and P as an array of ints.
// We implement the code to compute LU "in place" in the matrix A.
// In order to make some of the calculations more straight forward and to
// match Cormen's et al. pseudocode the matrix A should have its first row and first columns
// to be all 0.
func LUPDecomposition(A matrix) (matrix, []int) {

	n := len(A)
	// pi is the permutation matrix.
	// We implement it as an array whose value indicates which column the 1 would appear.
	//We use it to avoid dividing by zero or small numbers.
	pi := make([]int, n)
	var p float64
	var kp, pik, pikp int
	var aki, akpi float64

	for j := 0; j < n; j++ {
		pi[j] = j
	}

	for k := 0; k < n; k++ {
		p = 0
		for i := k; i < n; i++ {
			if math.Abs(A[i][k]) > p {
				p = math.Abs(A[i][k])
				kp = i
			}
		}
		if p == 0 {
			fmt.Println("Panic: singular matrix")
			panic(A)
		}

		pik = pi[k]
		pikp = pi[kp]
		pi[k] = pikp
		pi[kp] = pik

		for i := 0; i < n; i++ {
			aki = A[k][i]
			akpi = A[kp][i]
			A[k][i] = akpi
			A[kp][i] = aki
		}

		for i := k + 1; i < n; i++ {
			A[i][k] = A[i][k] / A[k][k]
			for j := k + 1; j < n; j++ {
				A[i][j] = A[i][j] - (A[i][k] * A[k][j])
			}
		}
	}
	return A, pi
}
