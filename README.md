# Alpaka expression trees as boost.odeint backend

This repo contains an implementation of expression trees which supports automatic device kernels fusion in a compile time.
The expressions based on Alpaka library and is being used to implement *boost.odeint* backend in order to use Alpaka
for solving initial value (Cauchy) problem.

*Note*: Initially it was a team project (in cooperation with Paul Hempel (TUD) and supervised by Dr. Jeffrey Kelling (HZDR)), but this is a copy
which contains a slightly modified code which was written by me (unless original author is specified in the header of the file).

## Alpaka

The Alpaka library is a header-only C++17 abstraction library for accelerator development. It abstract specific
accelerator API and allows user to write accelerated application for different types of devices from different manufactures.
For more information about Alpaka I recommend read the [official documentation](https://alpaka.readthedocs.io/en/latest/basic/intro.html).

This library provide an ability only to abstract from the specific accelerator, but one should still write their own kernels.

## Boost.Odeint

[Boost.Odeint](https://www.boost.org/doc/libs/1_84_0/libs/numeric/odeint/doc/html/index.html) is a library for solving initial value problems of ordinary differential equations. It abstracts actual calculation in a way
that anyone can add their own types for dynamic system state / derivatives and implement algebra and operations which will be used during integration.
More detailed requirements to new backends can be found in the official documentation: [here](https://www.boost.org/doc/libs/1_84_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/concepts/state_wrapper.html) and [here](https://www.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/concepts/state_algebra_operations.html).

## Eager and lazy evaluation

The classical approach to calculation we are used to is **eager**, i.e. if one applies an operation on a vector the result is immediately calculated.
This introduces additional memory allocations and kernels launches which can significantly sacrifice the performance. For example:
```[cpp]
vec a = ...;
vec b = ...;
vec c = ...;
vec res = a + b + c;
```
Here just the sum of 3 vector is calculated, but let's dig into it a little bit deeper.
Actually the computation in eager mode is performed like this:
```[cpp]
vec res_ab = a + b;
vec res = res_ab + c;
```
So, actually there were 2 kernel launches and 2 memory allocations.

The idea of **lazy evaluation** is to just gather information of all operands and operators (construct expression tree) and
evaluate the whole expression only once (where it is needed) and therefore allocate device memory and launch kernel only once.

## Explanation of the solution

### Boost.Odeint backend

Adding support for desired backend consists of implementing of 3 entities:
- State type / wrapper which wraps a state of the dynamic system. An expression `Vector` has been used as a state type and all required
functions like `is_resizeable`, `resize_impl::resize`, `same_size_impl::same_size`, default and copy ctors have been implemented for this type.
- Operations which abstracts operation on the state types. Already existent `default_operation` which requires only operators `+`, `abs`,
`/` and multiplication by scalar and reimplemented `ScaleSumSwap2` and `rel_error` has been reused
- Algebra which abstract way to call operations on the given states. Already existent `vector_space_algebra` which
just calls operations with given states as arguments has been reused.

### Expression trees

The expression system is mostly inspired by [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), but there are 3 additional difficulties:
1. Pointers to the device memory should be explicitly obtained and passed to the kernel
2. Operations should be known by device (and not all of them point wise)
3. Some expressions are not lazy evaluatable (e.g. reduction)

In short words the expression system works as follows: assume user wants to sum up 2 vectors: `a + b`. But the `operator+` instead of returning the value of type vector (which is also type of expression) returns specific `BinaryExpression` with a summation functor. Then user can use this `BinaryExpression`  to construct other expressions and form a tree of expressions. And actual fused calculation and allocation of memory for the result happens only on assignment to the resulting vector.

In order to solve problems described above the concept of the `AccExpressionHandler`s has been introduced.

![the concept of the `AccExpressionHandler`s](assets/acc_expr_handler_workflow.png)

In short words:
- When user calls assignment to `Vector` the evaluator class is used to compute the results.
- Evaluator gets a `AccExpressionHandler` from the root expression.
- Evaluator calls method `prepare()` of this handler. `prepare` method can internally call `prepare` methods of child nodes, get and store (for using during calculation) a pointer to the device memory or evaluate non-lazy expression and store the pointer to the device memory where the result of evaluation of non-lazy expression is placed. It solves 1 and 3 problems.
- Then evaluator launches the kernel and passes as an argument (by copying) the prepared root expression handler. The kernel just calls `get_value` method of the handler which recursively calls the `get_value` methods of the whole trees. And since the functor for calculating the result can be stored in the `AccExpressionHandler` a device knows how to compute the result. Therefore the problem 2 is solved.

So, we end up in only 1 fused kernel launch and only 1 memory allocation (for the result of the whole expression) for all our expression unless some of the expressions are non-lazy evaluatable (in this case the number of kernel launches and memory allocation will be increased by the number of non-lazy evaluatable expressions).

### Limitations
Since `get_value` method is called for all nodes except leaves, theoretically the kernel could run out of stack memory. But since usually functors and `get_value` implementations itself are very simple I **hope** that device compiler will inline them. If it didn't happen the only possible way to solve the problem is to split expression into two subexpressions and evaluate the first one by assigning to a `Vector`.

Also currently expressions support only 1 dimensional vectors, but they can be extended for arbitrary number of dimensions.

Since the expression trees are lazy, when one constructs an expression tree and then change one of operands (e.g. changed the first element), the result after assigning the tree to a `Vector` will be calculated using a changed operand.

The current implementation of assigning kernel is blocking (waits for completion of kernel), but theoretically it can be non-blocking if the user code guarantees that no one will use an internal `Vector`'s buffer from other queues (because otherwise it can end up in a race condition).
