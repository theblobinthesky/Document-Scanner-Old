for value in arm64-v8a armeabi-v7a x86 x86_64
do
    echo ""
    echo "Building android/$value"
    mkdir -p build/android/$value
    cd build/android/$value
    cmake -GNinja -DCMAKE_ANDROID_ARCH_ABI=$value -DTARGET_PLATFORM=Android ../../..
    ninja
    cd ../../..
done

for value in x86_64
do
    echo ""
    echo "Building linux/$value"
    mkdir -p build/linux/$value
    cd build/linux/$value
    cmake -GNinja -DTARGET_PLATFORM=Linux ../../..
    ninja
    cd ../../..
done