export BASE_DIR=/home/code/spn-compiler
cmake -GNinja -DCMAKE_PREFIX_PATH="$BASE_DIR/llvm-bin/lib/cmake/llvm;$BASE_DIR/llvm-bin/lib/cmake/mlir;$BASE_DIR/pybind11/install/share/cmake/pybind11;$BASE_DIR/spdlog/install/lib/cmake/spdlog;$BASE_DIR/capnproto/install"\
    -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_ASSERTIONS=ON\
    -DCMAKE_BUILD_TYPE=Debug\
    -DSPNC_BUILD_DOC=ON\
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON\
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..