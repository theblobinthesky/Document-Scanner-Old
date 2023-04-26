for value in arm64-v8a
do
    echo ""
    echo "Building $value"
    mkdir -p build/android/$value
    cd build/android/$value
    cmake -GNinja -DCMAKE_ANDROID_ARCH_ABI=$value -DTARGET_PLATFORM=Android -DCMAKE_BUILD_TYPE=$1 ../../..
    ninja
    cd ../../..
done