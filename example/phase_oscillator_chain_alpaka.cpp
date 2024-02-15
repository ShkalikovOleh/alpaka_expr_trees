// Adopted to our alpaka algebra thrust example

#include "algebra/alpaka.hpp"
#include "expressions/expressions.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>

#include <cmath>
#include <iostream>

using namespace std;

using namespace boost::numeric::odeint;

// change this to float if your device does not support double computation
typedef double value_type;
size_t const N = 32768;
value_type const pi = 3.1415926535897932384626433832795029;
value_type const epsilon = 6.0 / (N * N); // should be < 8/N^2 to see phase locking
value_type const dt = 0.1;

using Dim = alpaka::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
using BufAcc = alpaka::Buf<Acc, value_type, Dim, Idx>;
using state_type = alpaka_buffer_wrapper<BufAcc, QueueAcc, Acc>;
using DevHost = alpaka::DevCpu;
using BufHost = alpaka::Buf<DevHost, value_type, Dim, Idx>;

template<typename InnerExpr, int shift>
class ShiftExpression : public ExpressionBase<ShiftExpression<InnerExpr, shift>>
{
public:
    using value_type = typename expr_traits<ShiftExpression<InnerExpr, shift>>::value_type;
    using idx_type = typename expr_traits<ShiftExpression<InnerExpr, shift>>::idx_type;

public:
    struct AccExpressionHandler
    {
        using inner_handler = typename InnerExpr::AccExpressionHandler;
        inner_handler inner;
        std::size_t N;

        AccExpressionHandler(inner_handler const& inner, std::size_t N) : inner(inner), N(N)
        {
        }

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> value_type
        {
#pragma nv_diag_suppress 68
            idx_type real_idx = i + shift;
#pragma nv_diag_suppress 68
            if constexpr(shift < 0)
            {
                if(i < -shift)
                    real_idx = static_cast<idx_type>(0);
            }
            else
            {
                if(real_idx > N - 1)
                    real_idx = static_cast<idx_type>(N - 1);
            }
            return inner.getValue(real_idx);
        }

        void prepare()
        {
            inner.prepare();
        }
    };

private:
    InnerExpr expr_;
    std::size_t N_;
    int shift_;

public:
    ShiftExpression(InnerExpr const& expr) : expr_(expr)
    {
        this->queue_ = expr.getQueue();
        this->extent_ = expr.getExtent();
        N_ = this->extent_[0];
    }

    AccExpressionHandler getHandler() const
    {
        return {expr_.getHandler(), N_};
    }
};

template<typename InnerExpr, int shift>
struct expr_traits<ShiftExpression<InnerExpr, shift>>
{
    using acc_type = typename expr_traits<InnerExpr>::acc_type;
    using idx_type = typename expr_traits<InnerExpr>::idx_type;
    using dim_type = typename expr_traits<InnerExpr>::dim_type;
    using queue_type = typename expr_traits<InnerExpr>::queue_type;
    using value_type = typename expr_traits<InnerExpr>::value_type;
    using eval_ret_type = typename expr_traits<InnerExpr>::eval_ret_type;
    constexpr static bool is_binary_op = false;
    constexpr static bool is_lazy_evaluatable = true;
};

//<-
/*
 * This implements the rhs of the dynamical equation:
 * \phi'_0 = \omega_0 + sin( \phi_1 - \phi_0 )
 * \phi'_i  = \omega_i + sin( \phi_i+1 - \phi_i ) + sin( \phi_i - \phi_i-1 )
 * \phi'_N-1 = \omega_N-1 + sin( \phi_N-1 - \phi_N-2 )
 */
//->
class phase_oscillators
{
public:
    phase_oscillators(state_type const& omega) : m_omega(omega)
    {
    }

    void operator()(state_type const& x, state_type& dxdt, value_type const dt)
    {
        auto x_prev = ShiftExpression<state_type, -1>(x);
        auto x_next = ShiftExpression<state_type, 1>(x);
        dxdt = m_omega + sin(x_next - x) + sin(x - x_prev);
    }

private:
    state_type const& m_omega;
};
//]

template<typename TAcc>
state_type create_frequencies(std::size_t N, value_type epsilon, TAcc devAcc, QueueAcc queue, DevHost devHost)
{
    alpaka::Vec<Dim, Idx> extent{N};
    BufHost omegaHostBuf(alpaka::allocBuf<value_type, Idx>(devHost, extent));
    BufAcc omegaAccBuf(alpaka::allocBuf<value_type, Idx>(devAcc, extent));

    value_type* const omega(alpaka::getPtrNative(omegaHostBuf));
    for(size_t i = 0; i < N; ++i)
        omega[i] = (N - i) * epsilon;

    alpaka::memcpy(queue, omegaAccBuf, omegaHostBuf);
    return {queue, omegaAccBuf};
}

template<typename TAcc>
state_type get_initial_condition(std::size_t N, TAcc devAcc, QueueAcc queue, DevHost devHost)
{
    alpaka::Vec<Dim, Idx> extent{N};
    BufHost xHostBuf(alpaka::allocBuf<value_type, Idx>(devHost, extent));
    BufAcc xAccBuf(alpaka::allocBuf<value_type, Idx>(devAcc, extent));

    value_type* const x_host(alpaka::getPtrNative(xHostBuf));
    for(size_t i = 0; i < N; ++i)
        x_host[i] = 2.0 * pi * drand48();

    alpaka::memcpy(queue, xAccBuf, xHostBuf);
    return {queue, xAccBuf};
}

int main(int arc, char* argv[])
{
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    QueueAcc queue(devAcc);
    auto const devHost = alpaka::getDevByIdx<DevHost>(0u);

    auto omega = create_frequencies(N, epsilon, devAcc, queue, devHost);

    // create stepper
    runge_kutta4<state_type, value_type, state_type, value_type> stepper;

    // create phase oscillator system function
    phase_oscillators sys(omega);
    state_type x = get_initial_condition(N, devAcc, queue, devHost);

    // integrate
    integrate_const(stepper, sys, x, 0.0, 10.0, dt);

    alpaka::Vec<Dim, Idx> extent{N};
    BufHost xHostBuf(alpaka::allocBuf<value_type, Idx>(devHost, extent));
    alpaka::memcpy(queue, xHostBuf, x.getBuffer());
    alpaka::wait(queue);

    value_type* const x_host(alpaka::getPtrNative(xHostBuf));
    std::copy(x_host, x_host + N, std::ostream_iterator<value_type>(std::cout, "\n"));
    std::cout << std::endl;
    //]
}
