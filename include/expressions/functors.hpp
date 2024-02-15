#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

template<typename T1, typename T2>
struct AddFunctor
{
    using return_type = decltype(std::declval<T1&>() + std::declval<T2&>());

    static constexpr T1 identity = 0;

    ALPAKA_FN_ACC auto operator()(T1 a, T2 b) const -> return_type
    {
        return a + b;
    }
};

template<typename T1, typename T2>
struct DivisionFunctor
{
    using return_type = decltype(std::declval<T1>() / std::declval<T2>());

    static constexpr T1 identity = 1;

    ALPAKA_FN_ACC auto operator()(T1 a, T2 b) const -> return_type
    {
        return a / b;
    }
};

template<typename T1, typename T2>
struct SubFunctor
{
    using return_type = decltype(std::declval<T1>() - std::declval<T2>());

    static constexpr T1 identity = 0;

    ALPAKA_FN_ACC auto operator()(T1 a, T2 b) const -> return_type
    {
        return a - b;
    }
};

template<typename T1, typename T2>
struct MaxFunctor
{
    using return_type
        = std::remove_cv_t<std::remove_reference_t<decltype(std::max(std::declval<T1>(), std::declval<T2>()))>>;

    static constexpr T1 identity = std::numeric_limits<T1>::min();

    ALPAKA_FN_ACC auto operator()(T1 a, T2 b) const -> return_type
    {
        return std::max(a, b);
    }
};

template<typename TScalar, typename TExpr>
struct ScaleFunctor
{
    using return_type = decltype(std::declval<TScalar>() * std::declval<TExpr>());

    TScalar scalar;

    ScaleFunctor(TScalar scalar) : scalar(scalar)
    {
    }

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        return scalar * x;
    }
};

template<typename TExpr>
struct NegationFunctor
{
    using return_type = decltype(std::declval<TExpr>());

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        return -x;
    }
};

template<typename TScalar, typename TExpr>
struct AddScalarFunctor
{
    using return_type = decltype(std::declval<TScalar>() + std::declval<TExpr>());

    TScalar scalar;

    AddScalarFunctor(TScalar scalar) : scalar(scalar)
    {
    }

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        return scalar + x;
    }
};

template<typename TScalar, typename TExpr>
struct SubFromScalarFunctor
{
    using return_type = decltype(std::declval<TScalar>() - std::declval<TExpr>());

    TScalar scalar;

    SubFromScalarFunctor(TScalar scalar) : scalar(scalar)
    {
    }

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        return scalar - x;
    }
};

template<typename TExpr>
struct CosFunctor
{
    using return_type = decltype(std::cos(std::declval<TExpr>()));

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        using std::cos;
        return cos(x);
    }
};

template<typename TExpr>
struct SinFunctor
{
    using return_type = decltype(std::sin(std::declval<TExpr>()));

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        using std::sin;
        return sin(x);
    }
};

template<typename TExpr>
struct AbsFunctor
{
    using return_type = decltype(std::abs(std::declval<TExpr>()));

    ALPAKA_FN_ACC auto operator()(TExpr x) const -> return_type
    {
        using std::abs;
        return abs(x);
    }
};