#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

namespace py = pybind11;

class AdaptiveOscillator {
public:

    AdaptiveOscillator(int n_units, double dt,
                       const Eigen::VectorXd& w_scale, double alpha_lpf, bool F_input, bool normalize_vs, bool normalize_input)
        : n_units(n_units), dt(dt), 
        w_scale(w_scale), alpha_lpf(alpha_lpf), 
        F_input(F_input), normalize_vs(normalize_vs), normalize_input(normalize_input)
    {
        vs = Eigen::VectorXcd::Zero(n_units);
        ws = Eigen::VectorXd::Zero(n_units);
        spikes_out = Eigen::VectorXd::Zero(n_units);
        last_rot = Eigen::VectorXcd::Zero(n_units);
        approx_ws = Eigen::VectorXd::Zero(n_units);

        spike_threshold = 0.0;

        double start = 35.0;
        double stop  = 1450.0;

        if (n_units == 1) {
            ws[0] = start * 2.0 * M_PI;
        }
        else {
            double step = (stop - start) / (n_units - 1);

            for(int i=0;i<n_units;i++) {
                ws[i] = (start + step * i) * 2.0 * M_PI;
            }
        }

        // w_scale = lambda_scale;
        inv_w_scale = w_scale.cwiseInverse();

        for(int i=0;i<n_units;i++){
            last_rot[i] = std::exp(std::complex<double>(0.0f, dt * ws[i]));
            approx_ws[i] = ws[i];
        }

        // example: alpha_base = 5 Hz smoothing
        alpha_base = 5.0;

        ws_alpha.resize(n_units);

        // Precompute initial alpha values
        for (int i = 0; i < n_units; ++i) {
            double fcutoff = alpha_base + (ws[i] / (2*M_PI)) * 0.1; 
            double RC = 1.0 / (2.0 * M_PI * fcutoff);
            ws_alpha[i] = dt / (RC + dt);
        }
    }


    template<bool NormalizeVS, bool UseFInput, bool NormalizeInput>
    void updateSample(std::complex<double> sample)
    {
        std::complex<double> I(0,0);
        if(std::abs(sample.real()) > 0.0){
            I.real((sample.real() - vs.real().sum()) / n_units);
        } else {
            I.real(0.0);
        }
        if(std::abs(sample.imag()) > 0.0){
            I.imag((sample.imag() - vs.imag().sum()) / n_units);
        } else {
            I.imag(0.0);
        }

        for(int i=0; i<n_units; i++) {
            double w_dot = 0.0;
            std::complex<double>& input = UseFInput ? sample : I;
            
            if constexpr (NormalizeInput) {
                double magnitude = std::abs(input);
                
                if (magnitude > 0.0) {
                    double inv_mag = 1.0 / magnitude;
                    double input_real = input.real() * inv_mag;
                    double input_imag = input.imag() * inv_mag;

                    input.real(input_real);
                    input.imag(input_imag);
                }
            }

            if constexpr (!NormalizeVS) {
                w_dot = vs[i].imag() * input.real() - vs[i].real() * input.imag();
            }
            else {
                double magnitude = std::abs(vs[i]);
                
                if (magnitude > 0.0) {
                    double inv_mag = 1.0 / magnitude;
                    double vs_real = vs[i].real() * inv_mag;
                    double vs_imag = vs[i].imag() * inv_mag;

                    w_dot = vs_imag * input.real() - vs_real * input.imag();
                }
            }
            double w_update = w_dot * inv_w_scale[i];
            ws[i] -= w_update;

            vs[i] += input;
            if(spike_threshold > 0.0){
                double ws_diff = ws[i] - approx_ws[i];
                while(abs(ws_diff)>spike_threshold * 2.0 * M_PI){
                    double spike_sign = ws_diff > 0.0 ? 1.0 : -1.0;
                    approx_ws[i] += spike_sign *  spike_threshold * 2 * M_PI;
                    ws_diff = ws[i] - approx_ws[i];
                    spikes_out(hist_index, i) += spike_sign;
                }
            }
            double phase = dt * ws[i];
            last_rot[i] = { std::cos(phase), std::sin(phase) };
            vs[i] *= last_rot[i];
        }

        vs_hist.row(hist_index) = vs;
        ws_hist.row(hist_index) = ws;
        hist_index++;
    }

    template<bool NormalizeVS, bool UseFInput, bool NormalizeInput>
    void updateSequenceImpl(auto& buf, int N)
    {
        for(int i=0;i<N;i++) {
            updateSample<NormalizeVS, UseFInput, NormalizeInput>(buf(i));
        }
    }


    void updateSequence(py::array_t<std::complex<double>> sequence)
    {
        auto buf = sequence.unchecked<1>();

        int N = buf.shape(0);

        vs_hist.resize(N, n_units);
        ws_hist.resize(N, n_units);
        spikes_out.resize(N, n_units);

        vs_hist.setZero();
        ws_hist.setZero();
        spikes_out.setZero();

        hist_index = 0;

        if (normalize_vs && F_input && normalize_input)
            updateSequenceImpl<true, true, true>(buf, N);
        else if (normalize_vs && F_input && !normalize_input)
            updateSequenceImpl<true, true, false>(buf, N);
        else if (normalize_vs && !F_input && normalize_input)
            updateSequenceImpl<true, false, true>(buf, N);
        else if (normalize_vs && !F_input && !normalize_input)
            updateSequenceImpl<true, false, false>(buf, N);
        else if (!normalize_vs && F_input && normalize_input)
            updateSequenceImpl<false, true, true>(buf, N);
        else if (!normalize_vs && F_input && !normalize_input)
            updateSequenceImpl<false, true, false>(buf, N);
        else if (!normalize_vs && !F_input && normalize_input)
            updateSequenceImpl<false, false, true>(buf, N);
        else
            updateSequenceImpl<false, false, false>(buf, N);
    }


    py::array_t<std::complex<double>> get_vs_hist()
    {
        return eigenToNumpy(vs_hist);
    }

    py::array_t<double> get_ws_hist()
    {
        return eigenToNumpy(ws_hist);
    }

    py::array_t<double> get_spikes_out()
    {
        return eigenToNumpy(spikes_out);
    }

    void set_ws(py::array_t<double> new_ws)
    {
        auto buf = new_ws.unchecked<1>();

        for(int i=0; i<n_units; i++){
            ws[i] = buf(i);
            approx_ws[i] = buf(i);
        }
    }

    void set_vs(py::array_t<std::complex<double>> new_ws)
    {
        auto buf = new_ws.unchecked<1>();

        for(int i=0; i<n_units; i++){
            vs[i] = buf(i);
        }
    }

    void set_spike_threshold(double threshold)
    {
        spike_threshold = threshold;
    }

private:

    int n_units;
    bool F_input;
    bool normalize_vs;
    bool normalize_input;
    double dt;
    Eigen::VectorXd w_scale;
    Eigen::VectorXd inv_w_scale;
    double alpha_lpf;
    double spike_threshold;

    Eigen::VectorXcd vs;
    Eigen::VectorXd ws;

    Eigen::VectorXd ws_alpha; // per-oscillator smoothing factor
    double alpha_base;        // base smoothing factor (Hz) 

    Eigen::VectorXcd last_rot;
    Eigen::VectorXd approx_ws;

    Eigen::MatrixXcd vs_hist;
    Eigen::MatrixXd ws_hist;
    Eigen::MatrixXd spikes_out;

    int hist_index = 0;


    template<typename MatrixType>
    py::array_t<typename MatrixType::Scalar>
    eigenToNumpy(const MatrixType& mat)
    {
        using Scalar = typename MatrixType::Scalar;

        py::array_t<Scalar> array({mat.rows(), mat.cols()});
        auto buf = array.template mutable_unchecked<2>();

        for(int i = 0; i < mat.rows(); ++i)
            for(int j = 0; j < mat.cols(); ++j)
                buf(i,j) = mat(i,j);

        return array;
    }
};



PYBIND11_MODULE(adaptive_oscillator, m)
{
    py::class_<AdaptiveOscillator>(m, "AdaptiveOscillator")
        .def(py::init<int, double, const Eigen::VectorXd&, double, bool, bool, bool>(),
            py::arg("n_units"),
            py::arg("dt"),
            py::arg("lambda_scale"),
            py::arg("alpha_lpf"),
            py::arg("F_input"),
            py::arg("normalize_vs"),
            py::arg("normalize_input")
        )
        // .def("updateSample", &AdaptiveOscillator::updateSample)
        .def("updateSequence", &AdaptiveOscillator::updateSequence)
        .def("get_vs_hist", &AdaptiveOscillator::get_vs_hist)
        .def("get_ws_hist", &AdaptiveOscillator::get_ws_hist)
        .def("get_spikes_out", &AdaptiveOscillator::get_spikes_out)
        .def("set_ws", &AdaptiveOscillator::set_ws)
        .def("set_vs", &AdaptiveOscillator::set_vs)
        .def("set_spike_threshold", &AdaptiveOscillator::set_spike_threshold);
}