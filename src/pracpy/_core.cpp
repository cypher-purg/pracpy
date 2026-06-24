#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Crucial for std::vector conversion
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static py::array_t<double> make_array(std::size_t n) {
    return py::array_t<double>(n);
}

// ---------------------------------------------------------------------------
// single_decay — previously ra_decay
// ---------------------------------------------------------------------------
py::dict single_decay(double N0, double lam, double t_end, std::size_t points = 1000)
{
    if (N0 <= 0.0 || lam <= 0.0 || t_end <= 0.0 || points < 2)
        throw std::invalid_argument("Invalid inputs for single_decay");

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
// chain_decay — N-step radioactive decay chain (Bateman Equations)
// ---------------------------------------------------------------------------
py::dict chain_decay(double N0, const std::vector<double>& lambdas, double t_end, std::size_t points = 1000)
{
    std::size_t M = lambdas.size();
    if (M == 0) throw std::invalid_argument("lambdas list cannot be empty");
    if (N0 <= 0.0 || t_end <= 0.0 || points < 2) throw std::invalid_argument("Invalid simulation bounds");

    // The analytical Bateman formula divides by (lam_j - lam_i).
    // It breaks down if decay constants are identical.
    for (std::size_t i = 0; i < M; ++i) {
        if (lambdas[i] <= 0.0) throw std::invalid_argument("Lambdas must be positive");
        for (std::size_t j = i + 1; j < M; ++j) {
            if (std::abs(lambdas[i] - lambdas[j]) < 1e-12) {
                throw std::invalid_argument("Degenerate lambdas (identical decay constants) not supported in analytical solver.");
            }
        }
    }

    // Allocate a 1D array for time, and a 2D array for the isotope quantities
    py::array_t<double> t_arr(points);
    py::array_t<double> N_arr({M, points});

    // Use unchecked proxies to bypass bounds checking in the hot loop for raw speed
    auto t_buf = t_arr.mutable_unchecked<1>();
    auto N_buf = N_arr.mutable_unchecked<2>();

    // Pre-compute Bateman Coefficients (O(M^2)) outside the time loop
    // c[n][i] stores the coefficient for the i-th exponential term of the n-th isotope.
    std::vector<std::vector<double>> c(M, std::vector<double>(M, 0.0));

    for (std::size_t n = 0; n < M; ++n) {
        double prod_lambda = 1.0;
        for (std::size_t k = 0; k < n; ++k) {
            prod_lambda *= lambdas[k];
        }

        for (std::size_t i = 0; i <= n; ++i) {
            double denom = 1.0;
            for (std::size_t j = 0; j <= n; ++j) {
                if (i != j) {
                    denom *= (lambdas[j] - lambdas[i]);
                }
            }
            c[n][i] = prod_lambda / denom;
        }
    }

    // Hot Loop (O(Points * M))
    const double step = t_end / static_cast<double>(points - 1);
    std::vector<double> E(M); // Pre-allocated buffer for exp(-lambda * t)

    for (std::size_t pt = 0; pt < points; ++pt) {
        double t = step * static_cast<double>(pt);
        t_buf(pt) = t;

        // Compute all M exponentials exactly once per time step
        for (std::size_t i = 0; i < M; ++i) {
            E[i] = std::exp(-lambdas[i] * t);
        }

        // Apply pre-computed coefficients
        for (std::size_t n = 0; n < M; ++n) {
            double N_val = 0.0;
            for (std::size_t i = 0; i <= n; ++i) {
                N_val += c[n][i] * E[i];
            }
            N_buf(n, pt) = N0 * N_val;
        }
    }

    py::dict result;
    result["t"] = t_arr;
    result["N"] = N_arr;
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "practicals – C++ core for physics/chemistry simulations";

    m.def("single_decay", &single_decay,
          py::arg("N0"), py::arg("lam"), py::arg("t_end"), py::arg("points") = 1000,
          "Simulate simple exponential (radioactive) decay.");

    m.def("chain_decay", &chain_decay,
          py::arg("N0"), py::arg("lambdas"), py::arg("t_end"), py::arg("points") = 1000,
          "Simulate an N-step radioactive decay chain using a list of decay constants.");
}
