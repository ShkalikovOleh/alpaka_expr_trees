#pragma once

#include "state_wrapper.hpp"

#include <alpaka/alpaka.hpp>

#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

#include <type_traits>

namespace boost::numeric::odeint
{
    namespace detail
    {
        template<typename TScale1, typename TScale2>
        class ScaleSumSwap2Kernel
        {
        public:
            TScale1 a;
            TScale2 b;

            ScaleSumSwap2Kernel(TScale1 a1 = 1, TScale2 a2 = 1) : a(a1), b(a2)
            {
            }

            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TAcc, typename TElem1, typename TElem2, typename TElem3, typename TIdx>
            ALPAKA_FN_ACC auto operator()(
                TAcc const& acc,
                TElem1* const x1,
                TElem2* const x2,
                TElem3 const* const x3,
                TScale1 scale1,
                TScale2 scale2,
                TIdx const& numElements) const -> void
            {
                static_assert(alpaka::Dim<TAcc>::value == 1, "The ScaleSumSwap2Kernel expects 1-dimensional indices!");

                TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
                TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
                TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

                if(threadFirstElemIdx < numElements)
                {
                    TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
                    TIdx const threadLastElemIdxClipped(
                        (numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

                    for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
                    {
                        TElem1 tmp = x1[i];
                        x1[i] = a * x2[i] + b * x3[i];
                        x2[i] = tmp;
                    }
                }
            }
        };
    } // namespace detail

    struct alpaka_operations : default_operations
    {
        template<class Fac1 = double, class Fac2 = Fac1>
        struct scale_sum_swap2
        {
            Fac1 const a1;
            Fac2 const a2;

            scale_sum_swap2(Fac1 const alpha1, Fac2 const alpha2) : a1(alpha1), a2(alpha2)
            {
            }

            template<typename StateType1, typename StateType2, typename StateType3>
            void operator()(StateType1& x1, StateType2& x2, StateType3& x3) const
            {
                using Acc = typename StateType1::acc_type;
                auto queue = x1.getQueue();
                auto const devAcc = x1.getDevice();

                // Define the work division
                using Dim = alpaka::Dim<typename StateType1::buf_type>;
                using Idx = alpaka::Idx<typename StateType1::buf_type>;
                Idx const elementsPerThread(8u);
                alpaka::Vec<Dim, Idx> const extent = alpaka::getExtentVec(x1.getBuffer());

                // Let alpaka calculate good block and grid sizes given our full problem extent
                alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
                    devAcc,
                    extent,
                    elementsPerThread,
                    false,
                    alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

                detail::ScaleSumSwap2Kernel kernel{a1, a2};
                auto const taskKernel = alpaka::createTaskKernel<Acc>(
                    workDiv,
                    kernel,
                    alpaka::getPtrNative(x1.getBuffer()),
                    alpaka::getPtrNative(x2.getBuffer()),
                    alpaka::getPtrNative(x3.getBuffer()),
                    a1,
                    a2,
                    extent[0]);


                alpaka::enqueue(queue, taskKernel);
                alpaka::wait(queue);
            }
        };

        template<class Fac1 = double>
        struct rel_error
        {
            Fac1 const m_eps_abs, m_eps_rel, m_a_x, m_a_dxdt;

            rel_error(Fac1 const eps_abs, Fac1 const eps_rel, Fac1 const a_x, Fac1 const a_dxdt)
                : m_eps_abs(eps_abs)
                , m_eps_rel(eps_rel)
                , m_a_x(a_x)
                , m_a_dxdt(a_dxdt)
            {
            }

            template<typename StateType1, typename StateType2, typename StateType3>
            void operator()(StateType1& y, StateType2& x1, StateType3& x2) const
            {
                y = abs(y) / (m_eps_abs + m_eps_rel * (m_a_x * abs(x1) + m_a_dxdt * abs(x2)));
            }

            typedef void result_type;
        };
    };

    template<typename TBuf, typename TQueue, typename TAcc>
    struct operations_dispatcher<alpaka_buffer_wrapper<TBuf, TQueue, TAcc>>
    {
        typedef alpaka_operations operations_type;
    };

} // namespace boost::numeric::odeint