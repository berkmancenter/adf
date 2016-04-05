package adf

import (
	"math"
	"math/rand"
)

const seed = 101

type testCase struct {
	name     string
	skip     bool
	lag      int
	series   []float64
	pvalue   float64
	expected bool
}

var testCases = []testCase{
	{
		name:     "stationary",
		series:   genStationary(),
		pvalue:   -3.45,
		lag:      -1,
		expected: true,
	},
	{
		name:     "with linear trend",
		series:   genNonstationary(1),
		lag:      -1,
		pvalue:   -3.45,
		expected: false,
	},
	{
		name:     "with decreasing linear trend",
		series:   genNonstationary(-1),
		lag:      -1,
		pvalue:   -3.45,
		expected: false,
	},
	{
		name:     "stationary periodic",
		series:   genPeriodicStationary(),
		lag:      -1,
		pvalue:   -3.45,
		expected: true,
	},
	{
		name:     "periodic with linear trend",
		series:   genPeriodicNonstationary(),
		lag:      -1,
		pvalue:   -3.45,
		expected: false,
	},
	{
		name:     "with linear trend and outlier",
		series:   genNonstationaryWithOutlier(),
		pvalue:   -3.45,
		lag:      -1,
		expected: false,
	},
}

func genStationary() []float64 {
	rand.Seed(seed)
	series := make([]float64, 100)
	for i := range series {
		series[i] = rand.NormFloat64()
	}
	return series
}

func genNonstationary(b float64) []float64 {
	rand.Seed(seed)
	series := make([]float64, 100)
	for i := range series {
		series[i] = b*float64(i) + rand.NormFloat64()
	}
	return series
}

func genPeriodicStationary() []float64 {
	rand.Seed(seed)
	series := make([]float64, 100)
	for i := range series {
		series[i] = 2*math.Sin(float64(i)) + rand.NormFloat64()
	}
	return series
}

func genPeriodicNonstationary() []float64 {
	rand.Seed(seed)
	series := make([]float64, 100)
	for i := range series {
		series[i] = 5*math.Sin(float64(i)) - (0.5 * float64(i)) + rand.NormFloat64()
	}
	return series
}

func genNonstationaryWithOutlier() []float64 {
	rand.Seed(seed)
	series := make([]float64, 100)
	for i := range series {
		series[i] = float64(i) + rand.NormFloat64()
	}
	series[50] = 100
	return series
}
