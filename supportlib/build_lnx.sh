for value in x86_64
do
    echo ""
    echo "Building $value"
    mkdir -p build/linux/$value
    cd build/linux/$value
    cmake -GNinja -DTARGET_PLATFORM=Linux ../../..
    ninja
    cd ../../..
done