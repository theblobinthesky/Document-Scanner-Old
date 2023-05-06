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

if [ "$1" = "RUN" ]; then
    cd bin/linux/x86_64
    gdb -ex run --batch ./docscanner
    cd ../../..
fi