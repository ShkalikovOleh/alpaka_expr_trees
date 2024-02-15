// Credits: Paul Hempel

#include "algebra/alpaka.hpp"
#include "expressions/expressions.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <array>
#include <iostream>
#include <numeric>


auto main() -> int
{
    // Setup.
    using Idx = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Idx>;

    auto const dev = alpaka::getDevByIdx<Acc>(0);
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using Queue = alpaka::Queue<Acc, alpaka::NonBlocking>;
    Queue queue(dev);

    using Elem = int;

    // Create buffers.
    std::array<Elem, 5> inHost{1, 2, 3, 4, 5};

    auto inDev = alpaka::allocBuf<Elem, Idx>(dev, inHost.size());

    // Copy input to device.
    alpaka::memcpy(queue, inDev, inHost);
    alpaka::wait(queue);

    // Wrap the device buffers in an expression, then perform reduction.
    Vector<decltype(inDev), Queue, Acc> expr{queue, inDev};

    auto result = (expr + expr).sum().compute();

    std::cout << "Sum([1, 5] + [1, 5]) = " << result << ": ";

    if(result == 30)
    {
        std::cout << "\x1b[1;32mcorrect!\x1b[m\n";
    }
    else
    {
        std::cout << "\x1b[1;31mincorrect!\x1b[m\n";
    }
}