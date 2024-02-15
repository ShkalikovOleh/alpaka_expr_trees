#pragma once

#include "functors.hpp"

#include <alpaka/alpaka.hpp>

#include <memory>

template<typename TDerived>
class ExpressionBase;

template<typename TDerived>
struct expr_traits;

namespace impl_detail
{
    // modified copy from alpaka example

    template<typename TAcc>
    struct block_size_traits;

    template<typename TAcc, uint64_t TSize>
    static constexpr auto getMaxBlockSize() -> uint64_t
    {
        using traits = block_size_traits<TAcc>;
        return (traits::MaxBlockSize::value > TSize) ? TSize : traits::MaxBlockSize::value;
    }

    // Note: Boost Fibers, OpenMP 2 Threads and TBB Blocks accelerators aren't implented
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    //! OpenMP 2 Blocks defines
    //!
    //! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
    template<typename Dim, typename Idx>
    struct block_size_traits<alpaka::AccCpuOmp2Blocks<Dim, Idx>>
    {
        using MaxBlockSize = alpaka::DimInt<1u>;
    };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
    //! OpenMP 5 defines
    //!
    //! Defines Host, Device, etc. for the OpenMP 5 accelerator.
    template<typename Dim, typename Idx>
    struct block_size_traits<alpaka::AccOmp5<Dim, Idx>>
    {
        using MaxBlockSize = alpaka::DimInt<1u>;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    //! Serial CPU defines
    //!
    //! Defines Host, Device, etc. for the serial CPU accelerator.
    template<typename Dim, typename Idx>
    struct block_size_traits<alpaka::AccCpuSerial<Dim, Idx>>
    {
        using MaxBlockSize = alpaka::DimInt<1u>;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    //! CPU Threads defines
    //!
    //! Defines Host, Device, etc. for the CPU Threads accelerator.
    template<typename Dim, typename Idx>
    struct block_size_traits<alpaka::AccCpuThreads<Dim, Idx>>
    {
        using MaxBlockSize = alpaka::DimInt<1u>;
    };
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    //! CUDA defines
    //!
    //! Defines Host, Device, etc. for the CUDA/HIP accelerator.
    template<typename Dim, typename Idx>
    struct block_size_traits<alpaka::AccGpuCudaRt<Dim, Idx>>
    {
        using MaxBlockSize = alpaka::DimInt<1024u>;
    };
#endif

    template<typename T, uint64_t size>
    struct cheapArray
    {
        T data[size];

        //! Access operator.
        //!
        //! \param index The index of the element to be accessed.
        //!
        //! Returns the requested element per reference.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) -> T&
        {
            return data[index];
        }

        //! Access operator.
        //!
        //! \param index The index of the element to be accessed.
        //!
        //! Returns the requested element per constant reference.
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint64_t index) const -> T const&
        {
            return data[index];
        }
    };

    template<typename T, typename idx_type>
    struct DevicePointerAccExprHandler
    {
        using return_type = T;

        T* ptr_;

        DevicePointerAccExprHandler(T* ptr) : ptr_(ptr){};

        ALPAKA_FN_ACC auto getValue(idx_type i) -> T
        {
            return ptr_[i];
        }
    };

    //! A reduction kernel.
    //!
    //! \tparam TBlockSize The block size.
    //! \tparam T The data type.
    //! \tparam TFunc The Functor type for the reduction function.
    template<uint32_t TBlockSize, typename T, typename TFunc>
    struct ReduceKernel
    {
        ALPAKA_NO_HOST_ACC_WARNING

        //! The kernel entry point.
        //!
        //! \tparam TAcc The accelerator environment.
        //! \tparam TElem The element type.
        //! \tparam TIdx The index type.
        //!
        //! \param acc The accelerator object.
        //! \param source The source memory.
        //! \param destination The destination memory.
        //! \param n The problem size.
        //! \param func The reduction function.
        template<typename TAcc, typename TElem, typename TAccExprHandler, typename TIdx>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            TAccExprHandler handler,
            TElem* destination,
            TIdx const& n,
            TFunc func) const -> void
        {
            auto& sdata(alpaka::declareSharedVar<cheapArray<T, TBlockSize>, __COUNTER__>(acc));

            auto const blockIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));
            auto const threadIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]));
            auto const gridDimension(static_cast<uint32_t>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));

            // equivalent to blockIndex * TBlockSize + threadIndex
            auto const linearizedIndex(static_cast<uint32_t>(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]));

            T result = 0; // suppresses compiler warnings

            auto const gridSize = gridDimension * TBlockSize;
            auto dataIdx = linearizedIndex;

            if(threadIndex < n)
                result = handler.getValue(dataIdx); // avoids using the
                                                    // neutral element of specific

            // --------
            // Level 1: grid reduce, reading from global memory
            // --------

            // reduce per thread with increased ILP by 4x unrolling sum.
            // the thread of our block reduces its 4 grid-neighbored threads and
            // advances by grid-striding loop (maybe 128bit load improve perf)

            dataIdx += gridSize;
            while(dataIdx + 3 * gridSize < n)
            {
                result = func(
                    func(
                        func(result, func(handler.getValue(dataIdx), handler.getValue(dataIdx + gridSize))),
                        handler.getValue(dataIdx + 2 * gridSize)),
                    handler.getValue(dataIdx + 3 * gridSize));
                dataIdx += 4 * gridSize;
            }

            // doing the remaining blocks
            while(dataIdx < n)
            {
                result = func(result, handler.getValue(dataIdx));
                dataIdx += gridSize;
            }

            if(threadIndex < n)
                sdata[threadIndex] = result;

            alpaka::syncBlockThreads(acc);

            // --------
            // Level 2: block + warp reduce, reading from shared memory
            // --------

            ALPAKA_UNROLL()
            for(uint32_t currentBlockSize = TBlockSize,
                         currentBlockSizeUp = (TBlockSize + 1) / 2; // ceil(TBlockSize/2.0)
                currentBlockSize > 1;
                currentBlockSize = currentBlockSize / 2,
                         currentBlockSizeUp = (currentBlockSize + 1) / 2) // ceil(currentBlockSize/2.0)
            {
                bool cond
                    = threadIndex < currentBlockSizeUp // only first half of block
                                                       // is working
                      && (threadIndex + currentBlockSizeUp) < TBlockSize // index for second half must be in bounds
                      && (blockIndex * TBlockSize + threadIndex + currentBlockSizeUp) < n
                      && threadIndex < n; // if elem in second half has been initialized before

                if(cond)
                    sdata[threadIndex] = func(sdata[threadIndex], sdata[threadIndex + currentBlockSizeUp]);

                alpaka::syncBlockThreads(acc);
            }

            // store block result to gmem
            if(threadIndex == 0 && threadIndex < n)
                destination[blockIndex] = sdata[0];
        }
    };

    template<
        typename T,
        typename Idx,
        typename Dim,
        typename TAcc,
        typename DevAcc,
        typename QueueAcc,
        typename TAccExprHandler,
        typename TFunc>
    auto reduce(DevAcc devAcc, QueueAcc queue, Idx n, TAccExprHandler handler, TFunc func) -> T
    {
        using Extent = uint64_t;

        static constexpr uint64_t blockSize = getMaxBlockSize<TAcc, 256>();

        // calculate optimal block size (8 times the MP count proved to be
        // relatively near to peak performance in benchmarks)
        auto blockCount = static_cast<uint32_t>(alpaka::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount * 8);
        auto maxBlockCount = static_cast<uint32_t>((((n + 1) / 2) - 1) / blockSize + 1); // ceil(ceil(n/2.0)/blockSize)

        if(blockCount > maxBlockCount)
            blockCount = maxBlockCount;

        alpaka::Buf<DevAcc, T, Dim, Extent> destinationDeviceMemory
            = alpaka::allocBuf<T, Idx>(devAcc, static_cast<Extent>(blockCount));

        // create kernels with their workdivs
        ReduceKernel<blockSize, T, TFunc> kernel1, kernel2;
        alpaka::WorkDivMembers<Dim, Extent> workDiv1{
            static_cast<Extent>(blockCount),
            static_cast<Extent>(blockSize),
            static_cast<Extent>(1)};
        alpaka::WorkDivMembers<Dim, Extent> workDiv2{
            static_cast<Extent>(1),
            static_cast<Extent>(blockSize),
            static_cast<Extent>(1)};

        handler.prepare();

        // create main reduction kernel execution task
        auto const taskKernelReduceMain = alpaka::createTaskKernel<TAcc>(
            workDiv1,
            kernel1,
            handler,
            alpaka::getPtrNative(destinationDeviceMemory),
            n,
            func);

        DevicePointerAccExprHandler<T, Idx> ptrHandler{alpaka::getPtrNative(destinationDeviceMemory)};

        // create last block reduction kernel execution task
        auto const taskKernelReduceLastBlock = alpaka::createTaskKernel<TAcc>(
            workDiv2,
            kernel2,
            ptrHandler,
            alpaka::getPtrNative(destinationDeviceMemory),
            blockCount,
            func);

        // enqueue both kernel execution tasks
        alpaka::enqueue(queue, taskKernelReduceMain);
        alpaka::enqueue(queue, taskKernelReduceLastBlock);

        //  download result from GPU
        std::array<T, 1> resultGpuHost;
        alpaka::memcpy(queue, resultGpuHost, destinationDeviceMemory, static_cast<Extent>(1));
        alpaka::wait(queue);

        return resultGpuHost[0];
    }
} // namespace impl_detail

template<typename InnerExpr, typename Op>
class Reduction1DExpression : public ExpressionBase<Reduction1DExpression<InnerExpr, Op>>
{
public:
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename Op::return_type;
    using eval_ret_type = typename Op::return_type;
    using extent_type = typename alpaka::Vec<dim_type, idx_type>;

public:
    struct AccExpressionHandler
    {
        Reduction1DExpression const& results_;
        value_type reduction_res_;

        AccExpressionHandler(Reduction1DExpression const& results) : results_(results)
        {
        }

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> value_type
        {
            return reduction_res_;
        }

        void prepare()
        {
            reduction_res_ = results_.compute();
        }
    };

private:
    InnerExpr expr_;
    Op op_;

public:
    Reduction1DExpression(InnerExpr const& expr, Op const& op) : expr_(expr), op_(op)
    {
        this->queue_ = expr_.getQueue();
        this->extent_ = 1;
    }

    AccExpressionHandler getHandler() const
    {
        return {*this};
    }

    value_type compute() const
    {
        auto queue = expr_.getQueue();
        auto dev = alpaka::getDev(queue);
        auto const N = expr_.getExtent()[0];

        return impl_detail::reduce<value_type, idx_type, dim_type, acc_type>(dev, queue, N, expr_.getHandler(), op_);
        // return impl_detail::reduce<value_type>(dev);
    }
};

template<typename InnerExpr, typename Op>
struct expr_traits<Reduction1DExpression<InnerExpr, Op>>
{
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename Op::return_type;
    using eval_ret_type = typename Op::return_type;
    constexpr static bool is_binary_op = false;
    constexpr static bool is_lazy_evaluatable = false;
};