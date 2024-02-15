#pragma once

#include <alpaka/alpaka.hpp>

#include <memory>
#include <stdexcept>


template<typename TDerived>
class ExpressionBase;

template<typename TDerived>
struct expr_traits;

template<typename Lhs, typename Rhs, typename Functor>
class BinaryCwiseExpression : public ExpressionBase<BinaryCwiseExpression<Lhs, Rhs, Functor>>
{
public:
    using acc_type = typename Lhs::acc_type;
    using idx_type = typename Lhs::idx_type;
    using dim_type = typename Lhs::dim_type;
    using queue_type = typename Lhs::queue_type;
    using value_type = typename Functor::return_type;
    using extent_type = typename alpaka::Vec<dim_type, idx_type>;

public:
    struct AccExpressionHandler
    {
        using lexpr_handler = typename Lhs::AccExpressionHandler;
        using rexpr_handler = typename Rhs::AccExpressionHandler;

        Functor functor_;
        lexpr_handler lhs_;
        rexpr_handler rhs_;

        AccExpressionHandler(lexpr_handler lhs, rexpr_handler rhs, Functor functor)
            : lhs_{lhs}
            , rhs_{rhs}
            , functor_{functor} {};

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> typename Functor::return_type
        {
            return functor_(lhs_.getValue(i), rhs_.getValue(i));
        }

        void prepare()
        {
            lhs_.prepare();
            rhs_.prepare();
        }
    };

private:
    Functor functor_;
    Lhs lhs_;
    Rhs rhs_;

public:
    BinaryCwiseExpression(Lhs const& lhs, Rhs const& rhs, Functor functor) : functor_(functor), lhs_(lhs), rhs_(rhs)
    {
        this->queue_ = lhs.getQueue();

        auto lhs_extent = lhs.getExtent();
        auto rhs_extent = rhs.getExtent();
        if(lhs_extent[0] == 1)
            this->extent_ = rhs_extent;
        else if(rhs_extent[0] == 1)
            this->extent_ = lhs_extent;
        else if(lhs_extent == rhs_extent)
            this->extent_ = lhs_extent;
        else
            throw std::invalid_argument("Extents of arguments are mismatched");
    }

    AccExpressionHandler getHandler() const
    {
        return {lhs_.getHandler(), rhs_.getHandler(), functor_};
    }
};

template<typename Lhs, typename Rhs, typename Functor>
struct expr_traits<BinaryCwiseExpression<Lhs, Rhs, Functor>>
{
    using acc_type = typename Lhs::acc_type;
    using idx_type = typename Lhs::idx_type;
    using dim_type = typename Lhs::dim_type;
    using queue_type = typename Lhs::queue_type;
    using value_type = typename Functor::return_type;
    using eval_ret_type = typename expr_traits<Lhs>::eval_ret_type;
    constexpr static bool is_binary_op = true;

    // only for binary expressions
    constexpr static bool lhs_is_lazy_evaluatable = expr_traits<Lhs>::is_lazy_evaluatable;
    constexpr static bool rhs_is_lazy_evaluatable = expr_traits<Rhs>::is_lazy_evaluatable;
    // only for binary expressions

    constexpr static bool is_lazy_evaluatable = lhs_is_lazy_evaluatable & rhs_is_lazy_evaluatable;
};