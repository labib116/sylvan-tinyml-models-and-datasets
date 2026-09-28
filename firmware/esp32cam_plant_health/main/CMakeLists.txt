set(MODEL_FILE
    "${CMAKE_CURRENT_LIST_DIR}/../../../training_runs/plant_health_new_mobilenetv2_035/model_int8.tflite")
set(MODEL_GENERATOR
    "${CMAKE_CURRENT_LIST_DIR}/../tools/generate_model_data.py")
set(GENERATED_MODEL_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/model_data.cc")

if(NOT EXISTS "${MODEL_FILE}")
    message(FATAL_ERROR "Model not found: ${MODEL_FILE}")
endif()

idf_build_get_property(python PYTHON)

add_custom_command(
    OUTPUT "${GENERATED_MODEL_SOURCE}"
    COMMAND "${python}" "${MODEL_GENERATOR}"
            --input "${MODEL_FILE}"
            --output "${GENERATED_MODEL_SOURCE}"
    DEPENDS "${MODEL_FILE}" "${MODEL_GENERATOR}" "${CMAKE_CURRENT_LIST_DIR}/model_data.h"
    COMMENT "Embedding the quantized plant-health model"
    VERBATIM)

set_source_files_properties("${GENERATED_MODEL_SOURCE}" PROPERTIES GENERATED TRUE)

idf_component_register(
    SRCS "main.cc" "${GENERATED_MODEL_SOURCE}"
    INCLUDE_DIRS ".")

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
