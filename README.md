# SRGAN_tensorflow

DACON Super Resolution

```
python train.py --epoch --batch_size --save_path

optional arguments:
--epoch                   training epoch number
--batch_size              training image batch_size
--save_path               model save path ( srgan_{epoch}.h5 )
```

- train.py 데이터는 dacon train.csv 기준으로 작성
- 이미지 grid를 위해 단위 2**8 (256) 으로 defalut 설정 되어 있으며 hr 기준으로 설정
- 전체 이미지 블록은 HR / grid
- LR 이미지 (512, 512) 기준 64, 64  / HR 이미지 (2048, 2048) 기준 256,256 으로 crop 되게끔 지정

## HR
![image](https://user-images.githubusercontent.com/68021998/195014665-6e2b580d-44d0-46c2-8c00-0b4cca85f171.png)

## LR
![image](https://user-images.githubusercontent.com/68021998/195014724-f78192a7-39f1-44df-86da-b2970ace9b71.png)
