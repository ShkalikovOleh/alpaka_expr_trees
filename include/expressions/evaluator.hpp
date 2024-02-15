#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>

template<typename TDerived>
class ExpressionBase;

template<typename TBuf, typename TQueue, typename TAcc>
class Vector;

namespace impl_detail
{
    class AccExpressionHandlerKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TElem, typename TAccExprHandler, typename TIdx>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc, TElem* const res, TAccExprHandler expr, TIdx const& numElements)
            const -> void
        {
            static_assert(
                alpaka::Dim<TAcc>::value == 1,
                "The AccExpressionHandlerKernel expects 1-dimensional indices!");

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
                    res[i] = expr.getValue(i);
                }
            }
        }
    };

    template<typename TBuf, typename TQueue, typename TAcc, typename TExpr>
    void run_asign_kernel(Vector<TBuf, TQueue, TAcc>& res, TExpr& expr)
    {
        using Acc = TAcc;
        auto queue = res.getQueue();
        auto const devAcc = res.getDevice();

        // Define the work division
        using Dim = alpaka::Dim<TBuf>;
        using Idx = alpaka::Idx<TBuf>;
        Idx const elementsPerThread(8u);
        alpaka::Vec<Dim, Idx> const extent = alpaka::getExtentVec(res.getBuffer());

        // Let alpaka calculate good block and grid sizes given our full problem extent
        alpaka::WorkDivMembers<Dim, Idx> const workDiv(alpaka::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

        AccExpressionHandlerKernel kernel;
        auto handler = expr.getHandler();
        handler.prepare();
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            kernel,
            alpaka::getPtrNative(res.getBuffer()),
            handler,
            extent[0]);

        alpaka::enqueue(queue, taskKernel);
#ifndef NOT_WAIT_FOR_EXPR_EVAL
        alpaka::wait(queue);
#endif
    }
} // namespace impl_detail

template<
    typename TDerived,
    typename TOtherDerived,
    bool isLazyEvaluatable = expr_traits<TOtherDerived>::is_lazy_evaluatable>
struct evaluator;

template<typename TBuf, typename TQueue, typename TAcc, typename TOtherDerived>
struct evaluator<Vector<TBuf, TQueue, TAcc>, TOtherDerived, true>
{
    static Vector<TBuf, TQueue, TAcc>& assign(Vector<TBuf, TQueue, TAcc>& dest, TOtherDerived const& src)
    {
        auto extent = src.getExtent();
        auto queue = src.getQueue();
        dest.adjust_size(extent[0], queue);
        impl_detail::run_asign_kernel(dest, src);
        return dest;
    }
};

template<typename TBuf, typename TQueue, typename TAcc, typename TOtherDerived>
struct evaluator<Vector<TBuf, TQueue, TAcc>, TOtherDerived, false>
{
    static Vector<TBuf, TQueue, TAcc>& assign(Vector<TBuf, TQueue, TAcc>& dest, TOtherDerived const& src)
    {
        auto extent = src.getExtent();
        auto queue = src.getQueue();
        dest.adjust_size(extent[0], queue);
        impl_detail::run_asign_kernel(dest, src);
        return dest;
    }
};