for value in arm64-v8a armeabi-v7a x86 x86_64
do
    echo ""
    echo "Building $value"
    mkdir -p build/Android/$value
    cd build/Android/$value
    cmake -GNinja -DCMAKE_ANDROID_ARCH_ABI=$value -DTARGET_PLATFORM=Android ../../..
    ninja
    cd ../../..
done