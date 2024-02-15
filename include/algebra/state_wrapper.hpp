#pragma once

#include "../expressions/expressions.hpp"

#include <alpaka/alpaka.hpp>

#include <boost/numeric/odeint/util/copy.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/state_wrapper.hpp>

template<typename TBuf, typename TQueue, typename TAcc>
using alpaka_buffer_wrapper = Vector<TBuf, TQueue, TAcc>;


namespace boost::numeric::odeint
{
    template<typename TBuf, typename TQueue, typename TAcc>
    struct state_wrapper<alpaka_buffer_wrapper<TBuf, TQueue, TAcc>>
    {
        using state_type = alpaka_buffer_wrapper<TBuf, TQueue, TAcc>;

        state_type m_v;

        state_wrapper(){};
        state_wrapper(state_wrapper<state_type> const& other)
        {
            if(other.m_v.isInitialized())
            {
                auto buff = other.m_v.getBuffer();
                auto queue = other.m_v.getQueue();
                auto size = alpaka::getExtentVec(buff)[0];

                m_v.adjust_size(size, queue);

                alpaka::memcpy(queue, m_v.getBuffer(), buff);
            }
        }
    };

    template<typename TQueue, typename TAcc, typename TBuf1, typename TBuf2>
    struct same_size_impl<alpaka_buffer_wrapper<TBuf1, TQueue, TAcc>, alpaka_buffer_wrapper<TBuf2, TQueue, TAcc>>
    {
        static bool same_size(
            alpaka_buffer_wrapper<TBuf1, TQueue, TAcc> const& left,
            alpaka_buffer_wrapper<TBuf2, TQueue, TAcc> const& right)
        {
            if(left.isInitialized() && right.isInitialized())
            {
                return alpaka::getExtent<0>(left.getBuffer()) == alpaka::getExtent<0>(right.getBuffer());
            }
            else
            {
                return false;
            }
        }
    };

    template<typename TQueue, typename TAcc, typename TBuf>
    struct is_resizeable<alpaka_buffer_wrapper<TBuf, TQueue, TAcc>>
    {
        using type = boost::true_type;
        static bool const value = true;
    };

    template<typename TQueue, typename TAcc, typename TBuf1, typename TBuf2>
    struct resize_impl<alpaka_buffer_wrapper<TBuf1, TQueue, TAcc>, alpaka_buffer_wrapper<TBuf2, TQueue, TAcc>>
    {
        static void resize(
            alpaka_buffer_wrapper<TBuf1, TQueue, TAcc>& left,
            alpaka_buffer_wrapper<TBuf2, TQueue, TAcc> const& right)
        {
            auto size = alpaka::getExtent<0>(right.getBuffer());
            if(left.isInitialized())
            {
                left.adjust_size(size);
            }
            else
            {
                auto queue = right.getQueue();
                left.adjust_size(size, queue);
            }
        }
    };
} // namespace boost::numeric::odeint
