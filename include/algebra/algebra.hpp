#pragma once

#include "state_wrapper.hpp"

#include <boost/numeric/odeint.hpp>

namespace boost::numeric::odeint
{
    template<typename TBuf, typename TQueue, typename TAcc>
    struct vector_space_norm_inf<alpaka_buffer_wrapper<TBuf, TQueue, TAcc>>
    {
        using result_type = typename alpaka_buffer_wrapper<TBuf, TQueue, TAcc>::value_type;
        result_type operator()(alpaka_buffer_wrapper<TBuf, TQueue, TAcc> const& x) const
        {
            return x.max().compute();
        }
    };

    template<typename TBuf, typename TQueue, typename TAcc>
    struct algebra_dispatcher<alpaka_buffer_wrapper<TBuf, TQueue, TAcc>>
    {
        typedef vector_space_algebra algebra_type;
    };
} // namespace boost::numeric::odeint