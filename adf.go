// Package adf implements the Augmented Dickey-Fuller test.
package adf

import (
	"github.com/berkmancenter/ridger"
	"github.com/gonum/matrix/mat64"

	"math"
)

const (
	LPenalty      = 0.0001
	DefaultPValue = -3.45
)

type ADF struct {
	Series          []float64
	PValueThreshold float64
	Statistic       float64
}

// New creates and returns a new ADF test.
func New(series []float64, pvalue float64) *ADF {
	if pvalue == 0 {
		pvalue = DefaultPValue
	}
	return &ADF{Series: series, PValueThreshold: pvalue}
}

// Run runs the Augmented Dickey-Fuller test.
func (adf *ADF) Run() {
	series := adf.Series
	n := len(series) - 1
	lag := int(math.Floor(math.Cbrt(float64(n))))
	y := diff(series)
	k := lag + 1
	z := laggedMatrix(y, k)
	zcol1 := mat64.Col(nil, 0, z)
	xt1 := series[k:len(series)]
	trend := sequence(k, n)
	r, c := z.Dims()
	var design *mat64.Dense

	if k > 1 {
		yt1 := z.View(0, 1, r, c-1)
		design = mat64.NewDense(len(series)-1-k+1, 3+k-1, nil)
		design.SetCol(0, xt1)
		design.SetCol(1, ones(len(series)-1-k+1))
		design.SetCol(2, trend)

		_, c = yt1.Dims()
		for i := 0; i < c; i++ {
			design.SetCol(3+i, mat64.Col(nil, i, yt1))
		}
	} else {
		design = mat64.NewDense(len(series)-1-k+1, 3, nil)
		design.SetCol(0, xt1)
		design.SetCol(1, ones(len(series)-1-k+1))
		design.SetCol(2, trend)
	}

	rr := ridger.New(design, mat64.NewVector(len(zcol1), zcol1))
	rr.Regress(LPenalty)
	beta := rr.Coefficients
	sd := rr.StandardErrors

	adf.Statistic = beta[0] / sd[0]
}

func (adf ADF) IsStationary() bool {
	return adf.Statistic > adf.PValueThreshold
}

func (adf ADF) ZeroPaddedDiff() []float64 {
	d := make([]float64, len(adf.Series))
	d[0] = 0
	d = append(d, diff(adf.Series)...)
	return d
}

func diff(x []float64) []float64 {
	y := make([]float64, len(x)-1)
	for i := 0; i < len(x)-1; i++ {
		y[i] = x[i+1] - x[i]
	}
	return y
}

func laggedMatrix(series []float64, lag int) *mat64.Dense {
	r, c := len(series)-lag+1, lag
	m := mat64.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		for i := 0; i < c; i++ {
			m.Set(i, j, series[lag-j-1+i])
		}
	}
	return m
}

func sequence(start, end int) []float64 {
	seq := make([]float64, end-start+1)
	for i := start; i <= end; i++ {
		seq[i-start] = float64(i)
	}
	return seq
}

func ones(num int) []float64 {
	seq := make([]float64, num)
	for i := range seq {
		seq[i] = 1
	}
	return seq
}
