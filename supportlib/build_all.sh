for value in arm64-v8a armeabi-v7a x86 x86_64
do
    cd build/$value
    rm -rf *
    cmake -DCMAKE_ANDROID_ARCH_ABI=$value ../..
    make
    cd ../..
done