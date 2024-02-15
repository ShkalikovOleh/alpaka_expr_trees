#pragma once

#include "expression_base.hpp"

#include <alpaka/alpaka.hpp>

#include <optional>

template<typename TBuf, typename TQueue, typename TAcc>
class Vector : public ExpressionBase<Vector<TBuf, TQueue, TAcc>>
{
public:
    using acc_type = TAcc;
    using buf_type = TBuf;
    using queue_type = TQueue;
    using dim_type = alpaka::Dim<TBuf>;
    using idx_type = alpaka::Idx<TBuf>;
    using value_type = alpaka::Elem<TBuf>;

public:
    struct AccExpressionHandler
    {
        Vector const& vector_;
        value_type* ptr_;

        AccExpressionHandler(Vector const& vector) : vector_(vector)
        {
        }

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> value_type
        {
            return ptr_[i];
        }

        void prepare()
        {
            ptr_ = alpaka::getPtrNative(vector_.getBuffer());
        }
    };

private:
    // Internally this is a shared pointer.
    mutable std::optional<TBuf> buff_;

public:
    Vector() = default;

    Vector(TQueue& queue, TBuf& buffer) : buff_(buffer)
    {
        this->queue_ = queue;
        this->extent_ = alpaka::getExtentVec(buffer);
    }

    Vector(TQueue& queue, alpaka::Idx<TBuf> size = 1)
    {
        this->queue_ = queue;
        adjust_size(size);
    }

    template<typename TOtherDerived>
    inline Vector& operator=(ExpressionBase<TOtherDerived> const& other)
    {
        return ExpressionBase<Vector>::operator=(other.derived());
    }

    bool hasQueue() const
    {
        return bool(this->queue_);
    }

    bool hasBuffer() const
    {
        return bool(buff_);
    }

    bool isInitialized() const
    {
        return hasBuffer() && hasQueue();
    }

    AccExpressionHandler getHandler() const
    {
        return {*this};
    }

    TBuf& getBuffer() const
    {
        return *buff_;
    }

    alpaka::Dev<TBuf> getDevice() const
    {
        return alpaka::getDev(*buff_);
    }

    template<class TSize>
    void adjust_size(TSize new_size)
    {
        using value_type = alpaka::Elem<TBuf>;
        using idx_type = alpaka::Idx<TBuf>;

        if(static_cast<idx_type>(new_size) == this->extent_[0])
            return;
        this->extent_[0] = static_cast<idx_type>(new_size);

        auto dev = alpaka::getDev(*this->queue_);
        buff_ = alpaka::allocBuf<value_type, idx_type>(dev, this->extent_);
    }

    template<class TSize>
    void adjust_size(TSize new_size, TQueue& queue)
    {
        this->queue_ = queue;
        adjust_size(new_size);
    }
};

template<typename TBuf, typename TQueue, typename TAcc>
struct expr_traits<Vector<TBuf, TQueue, TAcc>>
{
    using acc_type = TAcc;
    using queue_type = TQueue;
    using dim_type = alpaka::Dim<TBuf>;
    using idx_type = alpaka::Idx<TBuf>;
    using value_type = alpaka::Elem<TBuf>;
    using eval_ret_type = Vector<TBuf, TQueue, TAcc>;
    constexpr static bool is_binary_op = false;
    constexpr static bool is_lazy_evaluatable = true;
};