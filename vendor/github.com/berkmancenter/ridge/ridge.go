// Package ridge implements ridge regression.
package ridge

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"

	"fmt"
	"math"
)

const BasicallyZero = 1.0e-15

func fmtMat(mat mat64.Matrix) fmt.Formatter {
	return mat64.Formatted(mat, mat64.Excerpt(2), mat64.Squeeze())
}

type RidgeRegression struct {
	X            *mat64.Dense
	XSVD         *mat64.SVD
	U            *mat64.Dense
	D            *mat64.Dense
	V            *mat64.Dense
	XScaled      *mat64.Dense
	Y            *mat64.Vector
	Scales       *mat64.Vector
	L2Penalty    float64
	Coefficients *mat64.Vector
	Fitted       []float64
	Residuals    []float64
	StdErrs      []float64
}

// New returns a new ridge regression.
func New(x *mat64.Dense, y *mat64.Vector, l2Penalty float64) *RidgeRegression {
	return &RidgeRegression{
		X:         x,
		Y:         y,
		L2Penalty: l2Penalty,
		Fitted:    make([]float64, y.Len()),
		Residuals: make([]float64, y.Len()),
	}
}

// Regress runs the ridge regression to calculate coefficients.
func (r *RidgeRegression) Regress() {
	r.scaleX()
	r.solveSVD()
	xr, _ := r.X.Dims()

	fitted := mat64.NewVector(xr, nil)
	fitted.MulVec(r.X, r.Coefficients)
	r.Fitted = fitted.RawVector().Data

	for i := range r.Residuals {
		r.Residuals[i] = r.Y.At(i, 0) - r.Fitted[i]
	}

	r.calcStdErr()
}

func (r *RidgeRegression) scaleX() {
	xr, xc := r.X.Dims()
	scaleData := make([]float64, xr)
	scalar := 1.0 / float64(xr)
	for i := range scaleData {
		scaleData[i] = scalar
	}
	scaleMat := mat64.NewDense(1, xr, scaleData)
	sqX := mat64.NewDense(xr, xc, nil)
	sqX.MulElem(r.X, r.X)

	scales := mat64.NewDense(1, xc, nil)
	scales.Mul(scaleMat, sqX)
	sqrtElem := func(i, j int, v float64) float64 { return math.Sqrt(v) }
	scales.Apply(sqrtElem, scales)
	r.Scales = mat64.NewVector(xc, scales.RawRowView(0))
	r.XScaled = mat64.NewDense(xr, xc, nil)
	scale := func(i, j int, v float64) float64 { return v / r.Scales.At(j, 0) }
	r.XScaled.Apply(scale, r.X)
}

func (r *RidgeRegression) solveSVD() {
	if r.XSVD == nil || r.XSVD.Kind() == 0 {
		r.XSVD = new(mat64.SVD)
		r.XSVD.Factorize(r.XScaled, matrix.SVDThin)
	}

	xr, xc := r.XScaled.Dims()
	xMinDim := int(math.Min(float64(xr), float64(xc)))

	u := mat64.NewDense(xr, xMinDim, nil)
	u.UFromSVD(r.XSVD)
	r.U = u

	s := r.XSVD.Values(nil)
	for i := 0; i < len(s); i++ {
		if s[i] < BasicallyZero {
			s[i] = 0
		} else {
			s[i] = s[i] / (s[i]*s[i] + r.L2Penalty)
		}
	}
	d := mat64.NewDense(len(s), len(s), nil)
	setDiag(d, s)
	r.D = d

	v := mat64.NewDense(xc, xMinDim, nil)
	v.VFromSVD(r.XSVD)
	r.V = v

	uty := mat64.NewVector(xMinDim, nil)
	uty.MulVec(u.T(), r.Y)

	duty := mat64.NewVector(len(s), nil)
	duty.MulVec(d, uty)

	coef := mat64.NewVector(xc, nil)
	coef.MulVec(v, duty)

	r.Coefficients = mat64.NewVector(xc, nil)
	r.Coefficients.DivElemVec(coef, r.Scales)
}

// http://stats.stackexchange.com/a/2126
// http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3228544/
func (r *RidgeRegression) calcStdErr() {
	xr, xc := r.X.Dims()
	xMinDim := int(math.Min(float64(xr), float64(xc)))
	errVari := 0.0
	for _, v := range r.Residuals {
		errVari += v * v
	}
	errVari /= float64(xr - xc)
	errVariMat := mat64.NewDense(xr, xr, nil)
	for i := 0; i < xr; i++ {
		errVariMat.Set(i, i, errVari)
	}

	//    V                   D                   UT        Z
	// xc x xMinDim * xMinDim x xMinDim * xMinDim x xr = xc x xr
	vd := mat64.NewDense(xc, xMinDim, nil)
	vd.Mul(r.V, r.D)
	z := mat64.NewDense(xc, xr, nil)
	z.Mul(vd, r.U.T())

	//    Z       ErrVar      ZT
	// xc x xr * xr x xr * xr x xc
	zerr := mat64.NewDense(xc, xr, nil)
	zerr.Mul(z, errVariMat)
	coefCovarMat := mat64.NewDense(xc, xc, nil)
	coefCovarMat.Mul(zerr, z.T())
	r.StdErrs = getDiag(coefCovarMat)
}

func getDiag(mat mat64.Matrix) []float64 {
	r, _ := mat.Dims()
	diag := make([]float64, r)
	for i := range diag {
		diag[i] = mat.At(i, i)
	}
	return diag
}

func setDiag(mat mat64.Mutable, d []float64) {
	for i, v := range d {
		mat.Set(i, i, v)
	}
}
