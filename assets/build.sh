mkdir -p build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=$1 ..
ninja
cd ..

if [ "$1" = "RUN" ]; then
    ./bin/packager
fi