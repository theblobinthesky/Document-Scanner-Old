mkdir -p build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=$1 ..
ninja
cd ..