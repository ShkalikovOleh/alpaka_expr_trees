// Adopted to our alpaka algebra thrust example

#include "algebra/alpaka.hpp"
#include "expressions/expressions.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/timer.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>

using namespace std;

using namespace boost::numeric::odeint;

// /*
//  * Sorry for that dirty hack, but nvcc has large problems with boost::random.
//  *
//  * Nevertheless we need the cauchy distribution from boost::random, and therefore
//  * we need a generator. Here it is:
//  */
struct drand48_generator
{
    typedef double result_type;
    result_type operator()(void) const
    {
        return drand48();
    }
    result_type min(void) const
    {
        return 0.0;
    }
    result_type max(void) const
    {
        return 1.0;
    }
};

// change this to float if your device does not support double computation
typedef double value_type;

using Dim = alpaka::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
using BufAcc = alpaka::Buf<Acc, value_type, Dim, Idx>;
using state_type = alpaka_buffer_wrapper<BufAcc, QueueAcc, Acc>;
using DevHost = alpaka::DevCpu;
using BufHost = alpaka::Buf<DevHost, value_type, Dim, Idx>;


struct mean_field_calculator
{
    static std::pair<value_type, value_type> get_mean(state_type const& x)
    {
        auto sin_sum = x.sin().sum().compute();
        auto cos_sum = x.cos().sum().compute();

        cos_sum /= value_type(x.getExtent()[0]);
        sin_sum /= value_type(x.getExtent()[0]);

        value_type K = sqrt(cos_sum * cos_sum + sin_sum * sin_sum);
        value_type Theta = atan2(sin_sum, cos_sum);

        return std::make_pair(K, Theta);
    }
};


class phase_oscillator_ensemble
{
public:
    phase_oscillator_ensemble(state_type const& omega, value_type epsilon = 1.0) : m_omega(omega), m_epsilon(epsilon)
    {
    }

    void set_epsilon(value_type epsilon)
    {
        m_epsilon = epsilon;
    }

    value_type get_epsilon(void) const
    {
        return m_epsilon;
    }

    void operator()(state_type const& x, state_type& dxdt, value_type const dt) const
    {
        auto const [K, Theta] = mean_field_calculator::get_mean(x);
        dxdt = m_omega + m_epsilon * K * sin(Theta - x);
    }

private:
    state_type m_omega;
    value_type m_epsilon;
};


struct statistics_observer
{
    value_type m_K_mean;
    size_t m_count;

    statistics_observer(void) : m_K_mean(0.0), m_count(0)
    {
    }

    template<class State>
    void operator()(State const& x, value_type t)
    {
        std::pair<value_type, value_type> mean = mean_field_calculator::get_mean(x);
        m_K_mean += mean.first;
        ++m_count;
    }

    value_type get_K_mean(void) const
    {
        return (m_count != 0) ? m_K_mean / value_type(m_count) : 0.0;
    }

    void reset(void)
    {
        m_K_mean = 0.0;
        m_count = 0;
    }
};


// const size_t N = 16384 * 128;
size_t const N = 16384;
value_type const pi = 3.1415926535897932384626433832795029;
value_type const dt = 0.1;
value_type const d_epsilon = 0.1;
value_type const epsilon_min = 0.0;
value_type const epsilon_max = 5.0;
value_type const t_transients = 10.0;
value_type const t_max = 100.0;

template<typename TAcc>
state_type create_frequencies(std::size_t N, value_type g, TAcc devAcc, QueueAcc queue, DevHost devHost)
{
    boost::cauchy_distribution<value_type> cauchy(0.0, g);
    drand48_generator d48;

    alpaka::Vec<Dim, Idx> extent{N};
    BufHost omegaHostBuf(alpaka::allocBuf<value_type, Idx>(devHost, extent));
    BufAcc omegaAccBuf(alpaka::allocBuf<value_type, Idx>(devAcc, extent));

    value_type* const omega(alpaka::getPtrNative(omegaHostBuf));
    for(size_t i = 0; i < N; ++i)
        omega[i] = cauchy(d48);

    alpaka::memcpy(queue, omegaAccBuf, omegaHostBuf);
    return {queue, omegaAccBuf};
}

BufHost get_initial_condition(std::size_t N, QueueAcc queue, DevHost devHost)
{
    alpaka::Vec<Dim, Idx> extent{N};
    BufHost xHostBuf(alpaka::allocBuf<value_type, Idx>(devHost, extent));

    value_type* const x_host(alpaka::getPtrNative(xHostBuf));
    for(size_t i = 0; i < N; ++i)
        x_host[i] = 2.0 * pi * drand48();

    return xHostBuf;
}

int main(int arc, char* argv[])
{
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    QueueAcc queue(devAcc);
    auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

    auto omegas = create_frequencies(N, 1.0, devAcc, queue, devHost);
    phase_oscillator_ensemble ensemble(omegas, 1.0);
    BufHost init = get_initial_condition(N, queue, devHost);

    boost::timer timer;
    boost::timer timer_local;
    double dopri5_time = 0.0, rk4_time = 0.0;
    {
        typedef runge_kutta_dopri5<state_type, value_type, state_type, value_type> stepper_type;

        ofstream fout("phase_ensemble_dopri5.dat");
        timer.restart();
        for(value_type epsilon = epsilon_min; epsilon < epsilon_max; epsilon += d_epsilon)
        {
            ensemble.set_epsilon(epsilon);
            statistics_observer obs;

            // copy to reuse the same initial condition
            state_type x{queue, N};
            alpaka::memcpy(queue, x.getBuffer(), init);

            timer_local.restart();

            // calculate some transients steps
            size_t steps1 = integrate_const(
                make_controlled(1.0e-6, 1.0e-6, stepper_type()),
                boost::ref(ensemble),
                x,
                0.0,
                t_transients,
                dt);

            // integrate and compute the statistics
            size_t steps2 = integrate_const(
                make_dense_output(1.0e-6, 1.0e-6, stepper_type()),
                boost::ref(ensemble),
                x,
                0.0,
                t_max,
                dt,
                boost::ref(obs));

            fout << epsilon << "\t" << obs.get_K_mean() << endl;
            cout << "Dopri5 : " << epsilon << "\t" << obs.get_K_mean() << "\t" << timer_local.elapsed() << "\t"
                 << steps1 << "\t" << steps2 << endl;
        }
        dopri5_time = timer.elapsed();
    }


    {
        typedef runge_kutta4<state_type, value_type, state_type, value_type> stepper_type;

        ofstream fout("phase_ensemble_rk4.dat");
        timer.restart();
        for(value_type epsilon = epsilon_min; epsilon < epsilon_max; epsilon += d_epsilon)
        {
            ensemble.set_epsilon(epsilon);
            statistics_observer obs;

            // copy to reuse the same initial condition
            state_type x{queue, N};
            alpaka::memcpy(queue, x.getBuffer(), init);

            timer_local.restart();

            // calculate some transients steps
            size_t steps1 = integrate_const(stepper_type(), boost::ref(ensemble), x, 0.0, t_transients, dt);

            // integrate and compute the statistics
            size_t steps2 = integrate_const(stepper_type(), boost::ref(ensemble), x, 0.0, t_max, dt, boost::ref(obs));
            fout << epsilon << "\t" << obs.get_K_mean() << endl;
            cout << "RK4     : " << epsilon << "\t" << obs.get_K_mean() << "\t" << timer_local.elapsed() << "\t"
                 << steps1 << "\t" << steps2 << endl;
        }
        rk4_time = timer.elapsed();
    }

    cout << "Dopri 5 : " << dopri5_time << " s\n";
    cout << "RK4     : " << rk4_time << "\n";

    return 0;
}
