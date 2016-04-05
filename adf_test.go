package adf

import (
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"

	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"testing"
)

func TestRun(t *testing.T) {
	for i, test := range testCases {
		if test.skip {
			continue
		}
		adf := New(test.series, test.pvalue, test.lag)
		adf.Run()
		observed := adf.IsStationary()
		//fmt.Printf("Expected: %v; Stat: %v; Lag: %v\n", test.expected, adf.Statistic, adf.Lag)

		if observed != test.expected {
			filename := "test_" + strconv.Itoa(i+1) + ".png"
			plotSeries(test.series, test.name, filename)
			csvf, err := os.Create("test_" + strconv.Itoa(i+1) + ".csv")
			if err != nil {
				panic(err)
			}
			w := csv.NewWriter(csvf)
			for _, v := range test.series {
				w.Write([]string{fmt.Sprint(v)})
			}
			w.Flush()
			t.Errorf("Failed %v. "+
				"Expected: %v but got %v with stat %v and p-value %v. See %v",
				test.name, test.expected, observed, adf.Statistic,
				adf.PValueThreshold, filename)
		}
	}
}

func plotSeries(series []float64, name, filename string) error {
	plot.DefaultFont = "Helvetica"
	p, err := plot.New()
	if err != nil {
		return err
	}

	p.Title.Text = name
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err = plotutil.AddLines(p, xys(series))
	if err != nil {
		return err
	}

	// Save the plot to a PNG file.
	if err := p.Save(6*vg.Inch, 4*vg.Inch, filename); err != nil {
		return err
	}
	return nil
}
func xys(series []float64) plotter.XYs {
	pts := make(plotter.XYs, len(series))
	for i, v := range series {
		pts[i].X = float64(i)
		pts[i].Y = v
	}
	return pts
}

func BenchmarkRun(b *testing.B) {
	test := testCases[4]
	for i := 0; i < b.N; i++ {
		adf := New(test.series, test.pvalue, test.lag)
		adf.Run()
	}
}
