#pragma once

#include <alpaka/alpaka.hpp>

#include <memory>

template<typename TDerived>
class ExpressionBase;

template<typename TDerived>
struct expr_traits;

template<typename InnerExpr, typename Functor>
class UnaryCwiseExpression : public ExpressionBase<UnaryCwiseExpression<InnerExpr, Functor>>
{
public:
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename Functor::return_type;
    using extent_type = typename alpaka::Vec<dim_type, idx_type>;

public:
    struct AccExpressionHandler
    {
        using expr_handler = typename InnerExpr::AccExpressionHandler;

        Functor functor_;
        expr_handler inner_;

        AccExpressionHandler(expr_handler inner, Functor functor) : inner_{inner}, functor_{functor} {};

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> typename Functor::return_type
        {
            return functor_(inner_.getValue(i));
        }

        void prepare()
        {
            inner_.prepare();
        }
    };

private:
    Functor functor_;
    InnerExpr expr_;

public:
    UnaryCwiseExpression(InnerExpr const& expr, Functor functor) : functor_(functor), expr_(expr)
    {
        this->queue_ = expr.getQueue();
        this->extent_ = expr.getExtent();
    }

    AccExpressionHandler getHandler() const
    {
        return {expr_.getHandler(), functor_};
    }
};

template<typename InnerExpr, typename Functor>
struct expr_traits<UnaryCwiseExpression<InnerExpr, Functor>>
{
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename Functor::return_type;
    using eval_ret_type = typename expr_traits<InnerExpr>::eval_ret_type;
    constexpr static bool is_binary_op = false;
    constexpr static bool is_lazy_evaluatable = true;
};