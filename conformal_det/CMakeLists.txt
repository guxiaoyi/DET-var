# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  # the ConformalDET class

  # Utility files
  cf_det_utils.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(conformal_det)
add_python_binding(conformal_det)
add_julia_binding(conformal_det)
add_go_binding(conformal_det)
add_r_binding(conformal_det)
add_markdown_docs(conformal_det "cli;python;julia;go;r" "misc. / other")
