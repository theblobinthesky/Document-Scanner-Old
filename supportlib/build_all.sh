for value in arm64-v8a armeabi-v7a x86 x86_64
do
    echo ""
    echo "Building $value"
    mkdir -p build/$value
    cd build/$value
    cmake -GNinja -DCMAKE_ANDROID_ARCH_ABI=$value ../..
    ninja
    cd ../..
done