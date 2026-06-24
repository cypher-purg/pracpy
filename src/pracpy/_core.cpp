#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static py::array_t<double> make_array(std::size_t n) {
    return py::array_t<double>(n);
}

// ---------------------------------------------------------------------------
// ra_decay — exponential (radioactive) decay
//
// Models:  N(t)     = N0 * exp(-lambda * t)   [remaining quantity]
//          decay(t) = N0 - N(t)               [decayed quantity]
//          t_half   = ln(2) / lambda           [half-life]
//
// Parameters
// ----------
// N0     : initial quantity (atoms, counts, mass …)
// lam    : decay constant  (lambda, s⁻¹ or any reciprocal time unit)
// t_end  : end time of the simulation
// points : number of sample points along [0, t_end]  (default 1000)
//
// Returns a Python dict with keys:
//   "t"      – time array  [0 … t_end]
//   "N"      – remaining quantity array
//   "decay"  – decayed quantity array
//   "lambda" – the decay constant used
//   "t_half" – computed half-life
// ---------------------------------------------------------------------------
py::dict ra_decay(double N0, double lam, double t_end,
                std::size_t points = 1000)
{
    if (N0 <= 0.0)
        throw std::invalid_argument("N0 must be positive (got "
                                    + std::to_string(N0) + ")");
    if (lam <= 0.0)
        throw std::invalid_argument("lambda must be positive (got "
                                    + std::to_string(lam) + ")");
    if (t_end <= 0.0)
        throw std::invalid_argument("t_end must be positive (got "
                                    + std::to_string(t_end) + ")");
    if (points < 2)
        throw std::invalid_argument("points must be >= 2");

    auto t_arr     = make_array(points);
    auto N_arr     = make_array(points);
    auto decay_arr = make_array(points);

    auto* t_buf     = t_arr.mutable_data();
    auto* N_buf     = N_arr.mutable_data();
    auto* decay_buf = decay_arr.mutable_data();

    const double step   = t_end / static_cast<double>(points - 1);
    const double t_half = std::log(2.0) / lam;

    for (std::size_t i = 0; i < points; ++i) {
        double t_i    = step * static_cast<double>(i);
        double N_i    = N0 * std::exp(-lam * t_i);
        t_buf[i]      = t_i;
        N_buf[i]      = N_i;
        decay_buf[i]  = N0 - N_i;
    }

    py::dict result;
    result["t"]      = t_arr;
    result["N"]      = N_arr;
    result["decay"]  = decay_arr;
    result["lambda"] = lam;
    result["t_half"] = t_half;
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "practicals – C++ core for physics/chemistry simulations";

    m.def("ra_decay", &ra_decay,
          py::arg("N0"),
          py::arg("lam"),
          py::arg("t_end"),
          py::arg("points") = 1000,
          R"doc(
Simulate exponential (radioactive) decay.

Parameters
----------
N0     : float
    Initial quantity (number of atoms, mass, counts, etc.)
lam    : float
    Decay constant λ (units: 1/time).  Related to half-life by
    λ = ln(2) / t_half.
t_end  : float
    End time of the simulation (same unit as 1/λ).
points : int, optional
    Number of evenly-spaced sample points (default 1000).

Returns
-------
dict with keys:
    "t"      – numpy array of time values [0 … t_end]
    "N"      – numpy array of remaining quantity  N(t) = N0·exp(−λt)
    "decay"  – numpy array of decayed quantity    N0 − N(t)
    "lambda" – the decay constant used
    "t_half" – computed half-life  ln(2)/λ
)doc");
}
