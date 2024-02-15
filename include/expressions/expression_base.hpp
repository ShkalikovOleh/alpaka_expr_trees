#pragma once

#include "1d_reduction.hpp"
#include "binary_cwise_expression.hpp"
#include "evaluator.hpp"
#include "functors.hpp"
#include "materialize_expression.hpp"
#include "unary_cwise_expression.hpp"

#include <alpaka/alpaka.hpp>

#include <optional>

template<typename TDerived>
struct expr_traits;

template<typename TDerived>
class ExpressionBase
{
public:
    // using handler_type = typename expr_traits<TDerived::AccExpressionHandler>;
    using acc_type = typename expr_traits<TDerived>::acc_type;
    using dim_type = typename expr_traits<TDerived>::dim_type;
    using idx_type = typename expr_traits<TDerived>::idx_type;
    using queue_type = typename expr_traits<TDerived>::queue_type;
    using value_type = typename expr_traits<TDerived>::value_type;
    using extent_type = typename alpaka::Vec<dim_type, idx_type>;

protected:
    std::optional<queue_type> queue_;
    extent_type extent_{0};

public:
    // every derived should implement
    // auto getHandler() const;

    TDerived const& derived() const
    {
        return *static_cast<TDerived const*>(this);
    }

    TDerived& derived()
    {
        return *static_cast<TDerived*>(this);
    }

    extent_type getExtent() const
    {
        return extent_;
    };

    queue_type getQueue() const
    {
        return *queue_;
    };

    template<typename TOtherDerived>
    inline TDerived& operator=(ExpressionBase<TOtherDerived> const& other)
    {
        return evaluator<TDerived, TOtherDerived>::assign(derived(), other.derived());
    }

    template<typename TOtherDerived>
    inline void eval_to(ExpressionBase<TOtherDerived>& dest)
    {
        return evaluator<TDerived, TOtherDerived>::assign(dest.derived(), derived());
    }

    template<typename Functor>
    inline UnaryCwiseExpression<TDerived, Functor> apply(Functor const& op) const
    {
        return {derived(), op};
    }

    template<typename Functor>
    inline Reduction1DExpression<TDerived, Functor> reduce(Functor const& op) const
    {
        return {derived(), op};
    }

    inline Reduction1DExpression<TDerived, AddFunctor<value_type, value_type>> sum() const
    {
        AddFunctor<value_type, value_type> op;
        return reduce(op);
    }

    inline Reduction1DExpression<TDerived, MaxFunctor<value_type, value_type>> max() const
    {
        MaxFunctor<value_type, value_type> op;
        return reduce(op);
    }

    inline UnaryCwiseExpression<TDerived, NegationFunctor<value_type>> operator-() const
    {
        return {derived(), NegationFunctor<value_type>{}};
    }

    inline UnaryCwiseExpression<TDerived, CosFunctor<value_type>> cos() const
    {
        return {derived(), CosFunctor<value_type>{}};
    }

    inline UnaryCwiseExpression<TDerived, SinFunctor<value_type>> sin() const
    {
        return {derived(), SinFunctor<value_type>{}};
    }

    inline UnaryCwiseExpression<TDerived, AbsFunctor<value_type>> abs() const
    {
        return {derived(), AbsFunctor<value_type>{}};
    }
};

template<typename TDerived, typename TOtherDerived>
inline BinaryCwiseExpression<
    TDerived,
    TOtherDerived,
    AddFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type>>
operator+(ExpressionBase<TDerived> const& lhs, ExpressionBase<TOtherDerived> const& rhs)
{
    AddFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type> functor;
    return {lhs.derived(), rhs.derived(), functor};
}

template<typename TDerived, typename TOtherDerived>
inline BinaryCwiseExpression<
    TDerived,
    TOtherDerived,
    DivisionFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type>>
operator/(ExpressionBase<TDerived> const& lhs, ExpressionBase<TOtherDerived> const& rhs)
{
    DivisionFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type>
        functor;
    return {lhs.derived(), rhs.derived(), functor};
}

template<typename TDerived, typename TOtherDerived>
inline BinaryCwiseExpression<
    TDerived,
    TOtherDerived,
    SubFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type>>
operator-(ExpressionBase<TDerived> const& lhs, ExpressionBase<TOtherDerived> const& rhs)
{
    SubFunctor<typename expr_traits<TDerived>::value_type, typename expr_traits<TOtherDerived>::value_type> functor;
    return {lhs.derived(), rhs.derived(), functor};
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, ScaleFunctor<TScalar, typename expr_traits<TDerived>::value_type>> operator*(
    ExpressionBase<TDerived> const& expr,
    TScalar const& scalar)
{
    ScaleFunctor<TScalar, typename expr_traits<TDerived>::value_type> functor{scalar};
    return {expr.derived(), functor};
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, ScaleFunctor<TScalar, typename expr_traits<TDerived>::value_type>> operator*(
    TScalar const& scalar,
    ExpressionBase<TDerived> const& expr)
{
    return expr * scalar;
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, AddScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type>> operator+(
    ExpressionBase<TDerived> const& expr,
    TScalar const& scalar)
{
    AddScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type> functor{scalar};
    return {expr.derived(), functor};
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, AddScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type>> operator+(
    TScalar const& scalar,
    ExpressionBase<TDerived> const& expr)
{
    return expr + scalar;
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, AddScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type>> operator-(
    ExpressionBase<TDerived> const& expr,
    TScalar const& scalar)
{
    AddScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type> functor{-scalar};
    return {expr.derived(), functor};
}

template<typename TDerived, typename TScalar, typename = std::enable_if_t<std::is_arithmetic_v<TScalar>>>
inline UnaryCwiseExpression<TDerived, SubFromScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type>>
operator-(TScalar const& scalar, ExpressionBase<TDerived> const& expr)
{
    SubFromScalarFunctor<TScalar, typename expr_traits<TDerived>::value_type> functor{scalar};
    return {expr.derived(), functor};
}

template<typename TDerived>
inline UnaryCwiseExpression<TDerived, CosFunctor<typename expr_traits<TDerived>::value_type>> cos(
    ExpressionBase<TDerived> const& expr)
{
    return expr.cos();
}

template<typename TDerived>
inline UnaryCwiseExpression<TDerived, SinFunctor<typename expr_traits<TDerived>::value_type>> sin(
    ExpressionBase<TDerived> const& expr)
{
    return expr.sin();
}

template<typename TDerived>
inline UnaryCwiseExpression<TDerived, AbsFunctor<typename expr_traits<TDerived>::value_type>> abs(
    ExpressionBase<TDerived> const& expr)
{
    return expr.abs();
}
