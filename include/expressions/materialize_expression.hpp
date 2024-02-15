#pragma once

#include <alpaka/alpaka.hpp>

#include <memory>

template<typename TDerived>
class ExpressionBase;

template<typename TDerived>
struct expr_traits;

template<typename InnerExpr>
class MaterializeExpression : public ExpressionBase<MaterializeExpression<InnerExpr>>
{
public:
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename InnerExpr::value_type;
    using extent_type = typename alpaka::Vec<dim_type, idx_type>;
    using eval_ret_type = typename expr_traits<InnerExpr>::eval_ret_type;

public:
    struct AccExpressionHandler
    {
        MaterializeExpression const& results_;
        value_type* ptr_;

        AccExpressionHandler(MaterializeExpression const& results) : results_(results)
        {
        }

        ALPAKA_FN_ACC auto getValue(idx_type i) const -> value_type
        {
            return ptr_[i];
        }

        void prepare()
        {
            results_.compute();
            ptr_ = results_.getPtr();
        }
    };

private:
    InnerExpr expr_;
    mutable eval_ret_type result_;

private:
    void compute() const
    {
        result_ = expr_;
    }

    auto getPtr() const
    {
        return alpaka::getPtrNative(result_.getBuffer());
    }

public:
    MaterializeExpression(InnerExpr const& expr) : expr_(expr)
    {
        this->queue_ = expr.getQueue();
        this->extent_ = expr.getExtent();
    }

    AccExpressionHandler getHandler() const
    {
        return {*this};
    }
};

template<typename InnerExpr>
struct expr_traits<MaterializeExpression<InnerExpr>>
{
    using acc_type = typename InnerExpr::acc_type;
    using idx_type = typename InnerExpr::idx_type;
    using dim_type = typename InnerExpr::dim_type;
    using queue_type = typename InnerExpr::queue_type;
    using value_type = typename InnerExpr::value_type;
    using eval_ret_type = typename expr_traits<InnerExpr>::eval_ret_type;
    constexpr static bool is_binary_op = false;
    constexpr static bool is_lazy_evaluatable = false;
};