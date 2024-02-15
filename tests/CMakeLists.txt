# Credits: Paul Hempel

function(create_test TEST_NAME TEST_SOURCES)
    alpaka_add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(
        ${TEST_NAME}
        PUBLIC alpaka::alpaka)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

create_test(algebra_test "algebra_test.cpp")
create_test(1d_reduction "1d_reduction.cpp")
