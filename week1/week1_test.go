package week1

import "testing"

func TestQ7(t *testing.T) {
	exp := q7()
	gotIterations := exp.avgIterations()
	if gotIterations < 7 || gotIterations > 22 {
		t.Errorf("week1 q7(): avgIterations == %s, want value in interval [7 : 22]", gotIterations)
	}
	gotDisagreement := exp.avgDisagreement()
	if gotDisagreement < 0.09 || gotDisagreement > 0.2 {
		t.Errorf("week1 q7(): avgDisagreement == %s, want value in interval [0.09 : 0.2]", gotIterations)
	}
}

func TestQ9(t *testing.T) {
	exp := q9()
	gotIterations := exp.avgIterations()
	if gotIterations > 115 || gotIterations < 90 {
		t.Errorf("week1 q9(): avgIterations == %s, want value in interval [90 : 115]", gotIterations)
	}
	gotDisagreement := exp.avgDisagreement()
	if gotDisagreement > 0.02 || gotDisagreement < 0.001 {
		t.Errorf("week1 q9(): avgDisagreement == %s, want value in interval [0.001 : 0.02]", gotIterations)
	}

}
