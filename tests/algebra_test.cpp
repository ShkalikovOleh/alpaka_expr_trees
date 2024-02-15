#include "algebra/alpaka.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <vector>

using namespace boost::numeric::odeint;

template<typename state_type>
struct harmonic_oscillator
{
    struct HarmonicOscillatorKernel
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TElem, typename TIdx>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            TElem const* const x,
            TElem* const dxdt,
            TIdx const& numElements) const -> void
        {
            static_assert(
                alpaka::Dim<TAcc>::value == 1,
                "The HarmonicOscillatorKernel expects 1-dimensional indices!");

            TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

            if(gridThreadIdx == 0)
                dxdt[0] = x[1];
            else if(gridThreadIdx == 1)
                dxdt[1] = -x[0] - 0.15 * x[1];
        }
    };

    void operator()(state_type const& x, state_type& dxdt, double const /* t */)
    {
        using Acc = typename state_type::acc_type;
        auto queue = x.getQueue();
        auto const devAcc = x.getDevice();

        // Define the work division
        using Dim = alpaka::Dim<typename state_type::buf_type>;
        using Idx = alpaka::Idx<typename state_type::buf_type>;
        Idx const elementsPerThread(1u);
        alpaka::Vec<Dim, Idx> const extent{2};

        // Let alpaka calculate good block and grid sizes given our full problem extent
        alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

        HarmonicOscillatorKernel kernel;
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            kernel,
            alpaka::getPtrNative(x.getBuffer()),
            alpaka::getPtrNative(dxdt.getBuffer()),
            extent[0]);

        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
    }
};

template<typename T>
struct harmonic_oscillator<std::vector<T>>
{
    void operator()(std::vector<T> const& x, std::vector<T>& dxdt, double const /* t */)
    {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - 0.15 * x[1];
    }
};

template<typename state_type>
struct push_back_state_and_time
{
    std::vector<state_type>& states_;
    std::vector<double>& times_;

    push_back_state_and_time(std::vector<state_type>& states, std::vector<double>& times)
        : states_(states)
        , times_(times)
    {
    }

    void operator()(state_type const& x, double t)
    {
        states_.push_back(x);
        times_.push_back(t);
    }
};

template<typename acc_state_type, typename host_buf_type>
struct copy_state_and_time
{
    std::vector<std::vector<typename acc_state_type::value_type>>& states_;
    host_buf_type& temp_host_buf_;
    typename acc_state_type::queue_type queue_;
    std::vector<double>& times_;

    copy_state_and_time(
        std::vector<std::vector<typename acc_state_type::value_type>>& states,
        std::vector<double>& times,
        host_buf_type& temp_host_buf,
        typename acc_state_type::queue_type queue)
        : states_(states)
        , times_(times)
        , temp_host_buf_(temp_host_buf)
        , queue_(queue)
    {
    }

    void operator()(acc_state_type const& x, double t)
    {
        auto size = alpaka::getExtentVec(x.getBuffer())[0];
        alpaka::memcpy(queue_, temp_host_buf_, x.getBuffer());
        auto* const state = alpaka::getPtrNative(temp_host_buf_);

        std::vector<typename acc_state_type::value_type> copy(size);
        std::copy(state, state + size, copy.begin());

        states_.push_back(copy);
        times_.push_back(t);
    }
};

auto is_float_equal(float a, float b) -> bool
{
    return fabs(a - b) < 10e-2;
}

auto main() -> int
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    using Data = std::float_t;
    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    using DevHost = alpaka::DevCpu;
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    using acc_state_type = alpaka_buffer_wrapper<BufAcc, QueueAcc, Acc>;
    using host_state_type = std::vector<Data>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    QueueAcc queue(devAcc);

    Idx const numElements(2);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    auto const devHost = alpaka::getDevByIdx<DevHost>(0u);
    BufHost bufHostX(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostTemp(alpaka::allocBuf<Data, Idx>(devHost, extent));
    Data* const pBufHostX(alpaka::getPtrNative(bufHostX));
    pBufHostX[0] = 1;
    pBufHostX[1] = 0;

    BufAcc bufAccX(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    alpaka::memcpy(queue, bufAccX, bufHostX);

    acc_state_type xAcc{queue, bufAccX};
    harmonic_oscillator<acc_state_type> systemAcc;
    runge_kutta4_classic<acc_state_type> stepperAcc;
    std::vector<host_state_type> statesAcc;
    std::vector<double> times;
    copy_state_and_time<acc_state_type, BufHost> observerAcc{statesAcc, times, bufHostTemp, queue};
    integrate_const(stepperAcc, systemAcc, xAcc, 0.0, 10.0, 0.1, observerAcc);

    host_state_type xHost = {pBufHostX[0], pBufHostX[1]};
    harmonic_oscillator<host_state_type> systemHost;
    runge_kutta4_classic<host_state_type> stepperHost;
    std::vector<host_state_type> statesHost;
    push_back_state_and_time<host_state_type> observerHost{statesHost, times};
    integrate_const(stepperHost, systemHost, xHost, 0.0, 10.0, 0.1, observerHost);

    for(int i = 0; i < statesAcc.size(); ++i)
    {
        host_state_type const& stateAcc = statesAcc[i];
        host_state_type const& stateHost = statesHost[i];

        std::cout << "ACC  "
                  << "t: " << times[i] << "\tx: " << stateAcc[0] << "\tv: " << stateAcc[1] << std::endl;
        std::cout << "HOST "
                  << "t: " << times[i] << "\tx: " << stateHost[0] << "\tv: " << stateHost[1] << std::endl;
        if(!is_float_equal(stateAcc[0], stateHost[0]))
            return 1;
    }

    std::cout << "Execution results correct!" << std::endl;
    return 0;
}