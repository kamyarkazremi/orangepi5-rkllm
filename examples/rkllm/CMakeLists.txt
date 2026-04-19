if (LLAMA_RKLLM)
    add_executable(rkllm-main main.cpp)

    target_include_directories(rkllm-main PRIVATE
        ${CMAKE_SOURCE_DIR}           # for ggml-rkllm.h, ggml.h, ggml-backend.h
        ${CMAKE_SOURCE_DIR}/common    # for common.h, sampling.h, etc.
        ${RKLLM_INCLUDE_DIRS}         # set by root CMakeLists from RKLLM_ROOT
    )

    target_link_libraries(rkllm-main PRIVATE
        ggml-rkllm                    # the backend static lib defined in root CMakeLists
        ${RKLLM_LIBRARIES}            # librkllmrt.so
        Threads::Threads
    )

    # Tell the linker where to find librkllmrt.so at runtime on the Orange Pi 5.
    # Adjust this path to match where you installed the RKLLM runtime.
    set_target_properties(rkllm-main PROPERTIES
        INSTALL_RPATH "${RKLLM_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    install(TARGETS rkllm-main RUNTIME DESTINATION bin)
endif()
