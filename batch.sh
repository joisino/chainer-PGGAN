if [ $# != 1 ]; then
    echo "Usage: $0 epoch"
    exit 1
fi

./train.py -g 0 --epoch $1 --depth 0
cp -r results results_depth0
cp -r img img_depth0

for i in `seq 1 6`; do
    ./train.py -g 0 --gen results/gen --dis results/dis --optg results/opt_g --optd results/opt_d --epoch $1 --depth $i
    cp -r results results_depth$i
    cp -r img img_depth$i
done
