package main

import "testing"

func TestQ7(t *testing.T) {
	exp := week1Q7()
	gotIterations := exp.avgIterations()
	if gotIterations < 7 || gotIterations > 22 {
		t.Errorf("wee1Q7(): avgIterations == %s, want value in interval [7 : 22]", gotIterations)
	}
	gotDisagreement := exp.avgDisagreement()
	if gotDisagreement < 0.09 || gotDisagreement > 0.2 {
		t.Errorf("wee1Q7(): avgDisagreement == %s, want value in interval [0.09 : 0.2]", gotIterations)
	}
}

func TestQ9(t *testing.T) {
}
