// Package adf implements the Augmented Dickey-Fuller test.
package adf

import (
	"github.com/berkmancenter/ridge"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"

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
	Lag             int
}

// New creates and returns a new ADF test.
func New(series []float64, pvalue float64, lag int) *ADF {
	if pvalue == 0 {
		pvalue = DefaultPValue
	}
	if lag < 0 {
		lag = int(math.Floor(math.Cbrt(float64(len(series)))))
	}
	newSeries := make([]float64, len(series))
	copy(newSeries, series)
	return &ADF{Series: newSeries, PValueThreshold: pvalue, Lag: lag}
}

// Run runs the Augmented Dickey-Fuller test.
func (adf *ADF) Run() {
	series := adf.Series
	mean := stat.Mean(series, nil)
	if mean != 0.0 {
		for i, v := range series {
			series[i] = v - mean
		}
	}
	n := len(series) - 1
	y := diff(series)
	lag := adf.Lag
	k := lag + 1
	z := laggedMatrix(y, k)
	zcol1 := mat64.Col(nil, 0, z)
	xt1 := series[k-1 : n]
	r, c := z.Dims()
	var design *mat64.Dense

	if k > 1 {
		yt1 := z.View(0, 1, r, c-1)
		design = mat64.NewDense(n-k+1, k, nil)
		design.SetCol(0, xt1)

		_, c = yt1.Dims()
		for i := 0; i < c; i++ {
			design.SetCol(1+i, mat64.Col(nil, i, yt1))
		}
	} else {
		design = mat64.NewDense(n-k+1, 1, nil)
		design.SetCol(0, xt1)
	}
	regressY := mat64.NewVector(len(zcol1), zcol1)

	rr := ridge.New(design, regressY, LPenalty)
	rr.Regress()
	beta := rr.Coefficients.RawVector().Data
	sd := rr.StdErrs

	adf.Statistic = beta[0] / sd[0]
}

func (adf ADF) IsStationary() bool {
	return adf.Statistic < adf.PValueThreshold
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
		for i := 0; i < r; i++ {
			m.Set(i, j, series[lag-j-1+i])
		}
	}
	return m
}
