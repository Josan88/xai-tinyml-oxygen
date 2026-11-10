cmake_minimum_required(VERSION 3.14)
project(fyrp_xai_tinyml_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../firmware/Arduino Nano 33/include")

add_executable(test_runner test_oxygen_risk_pipeline.cpp)
add_executable(stress_runner stress_test_oxygen_risk_pipeline.cpp)
add_executable(probability_probe probability_probe.cpp)
add_executable(context_probe context_probe.cpp)
add_executable(feature_audit_probe feature_audit_probe.cpp)
add_executable(qualification_probe qualification_probe.cpp)
add_executable(timing_pipeline_test test_timing_pipeline.cpp)

if(MSVC)
    target_compile_options(test_runner PRIVATE /W4 /EHsc)
    target_compile_options(stress_runner PRIVATE /W4 /EHsc)
else()
    target_compile_options(test_runner PRIVATE -Wall -Wextra)
    target_compile_options(stress_runner PRIVATE -Wall -Wextra)
endif()

enable_testing()
add_test(NAME TimingPipelineRegression COMMAND timing_pipeline_test)
add_test(NAME OxygenRiskPipelineTest COMMAND test_runner)
add_test(NAME OxygenRiskPipelineStressTest COMMAND stress_runner)

# Opt in with an explicit Python interpreter containing NumPy and SciPy.
# This regression uses checked-in windows and does not require outputs/ or data/.
set(PARITY_PYTHON "" CACHE FILEPATH "Python interpreter for independent SciPy parity regression")
if(PARITY_PYTHON)
    add_test(NAME TimingBoardProtocol COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_timing_board_protocol.py")
    add_test(NAME QualificationProbeRegression COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_qualification_probe.py"
        --probe $<TARGET_FILE:qualification_probe>)
    add_test(NAME FrozenNumericalCriteria COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_parity_criteria.py")
    add_test(NAME FeatureAuditRegression COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_feature_audit.py"
        --audit-probe $<TARGET_FILE:feature_audit_probe>
        --context-probe $<TARGET_FILE:context_probe>)
    add_test(NAME ProbabilityRegression COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_probability_regression.py"
        --probe $<TARGET_FILE:probability_probe>)
    add_test(NAME ContextPipelineRegression COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_context_pipeline.py"
        --probe $<TARGET_FILE:context_probe>)
    add_test(NAME BoardReplayProtocol COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_board_replay_protocol.py")
    add_test(NAME ContextBoardProtocol COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_context_board_protocol.py")
    add_test(NAME RecordedBoardProtocol COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_recorded_board_protocol.py")
    add_test(NAME ContextStateRegression COMMAND "${PARITY_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/replay_context_board.py"
        --probe $<TARGET_FILE:context_probe>)
endif()
