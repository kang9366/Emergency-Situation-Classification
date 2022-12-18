## 2.1 사용 데이터

- 데이터셋 출처 : [https://commonvoice.mozilla.org/](https://commonvoice.mozilla.org/)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0afa09d-22aa-4174-a2bf-9f102488de49/Untitled.png)

### Train 데이터

- 아프리카, 호주, 캐나다, 영국, 홍콩, 미국의 총 6개국의 화자의 문장 녹음 (.wav파일) 데이터셋
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/77a5bfc7-204e-44be-b957-2630e22cf816/Untitled.png)
    
- 클래스 별 1000개의 데이터
    
    : 더 많은 데이터 셋이 있었지만, 메모리 문제와 class imbalance 문제를 피하기 위하여 각 class별 데이터를 1000개로 통일시켰습니다.
    
- X_train = 6개국의 wav 파일
- y_train = 6개국의 label (아프리카부터 미국 순으로 0,1,2,3,4,5,6) : 직접 label
    - 아프리카
        
        [common_voice_en_19493.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2172fb7-2e6b-41c5-b48a-f8bab72aaf08/common_voice_en_19493.wav)
        
    - 호주
        
        [common_voice_en_337480.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0311356c-e85a-4c5f-ae36-fafc98159b7a/common_voice_en_337480.wav)
        
    - 캐나다
        
        [common_voice_en_155650.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b580fa03-e206-4002-bd15-5c6bfab117ae/common_voice_en_155650.wav)
        
    - 영국
        
        [common_voice_en_18133814.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b18a6e4-7d41-4a3f-be86-a55d0c12e94b/common_voice_en_18133814.wav)
        
    - 홍콩
        
        [common_voice_en_23373964.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e7fca73-48ed-41e7-8375-018ba842a5c8/common_voice_en_23373964.wav)
        
    - 미국
        
        [common_voice_en_17765621.wav](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8df5ea44-58bb-4294-a9e0-cbceff97afb9/common_voice_en_17765621.wav)
        

### Test 데이터

: 총 6개국의 화자의 문장 녹음 데이터 셋 파일 1000개 (.wav)

## 2.3 데이터 전처리

### 전체 과정

1. librosa 모듈 사용하여 wav파일 로드
    - wav파일을 train시 불러오는 과정이 오래 걸리므로, load가 빠른 npy파일로 train 데이터를  저장하였습니다.
    - 메모리 문제로 인하여, load시 형태를 float32로 지정해주었습니다.
    - train set에서 같은 사람이 여러 번 (평균 3번) 녹음 한 것을 확인하고, sort하여 같은사람이 녹음한 것들을 번호를 붙여서 불러왔습니다. 추후 학습 과정에서 3번씩 녹음한 것에 대한 index를 처리해주었습니다.
2. Melspectrogram 변환 후 , librosa.power_to_db 라이브러리 사용
3. 데이터 값의 범위를 균일하게 만들어 주기 위하여 scaling을 적용하였는데 min-max와 standardization 두가지 방식으로 scaling하였습니다.
4. 전처리를 마친 파일을 npy파일로 저장
    
    : 마찬가지로, 메모리 문제로 인하여 정규화까지 진행한 data를 npy파일로 저장해놓고 불러와 사용하였습니다.
    

### Mel-Spectrogram

- 입력 신호(음성 파일)을 시간 단위로 쪼개어, 다양한 주파수를 가지는 주기함수로 분해하고, 사람이 더 예민하게 인식하는 저주파 부분의 해상력을 높인 mel scale로 변환해 주는 과정입니다.

![불러온 wav를 librosa.load를 통해 불러오면 위의 결과와 같이 sampling rate(sr) 만큼의 float 값을 가지게 됩니다](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86725337-4ea5-43d0-afae-12c9823f0355/Untitled.png)

불러온 wav를 librosa.load를 통해 불러오면 위의 결과와 같이 sampling rate(sr) 만큼의 float 값을 가지게 됩니다

![Mel spectrogram 변환 결과(log scale)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c477729b-e2f0-4b22-baf4-e68442b6e3f5/Untitled.png)

Mel spectrogram 변환 결과(log scale)

- Arguments
    - sr(sampling rate) : 초당 sample의 개수. 데이터셋 wav파일의 경우엔 16000
    - n_fft(=win_length) : 음성을 얼마만큼의 길이로 자를 것인지
    - hop_length : 음성의 magnitude를 얼만큼 겹친 상태로 잘라서 보여줄 것인지
    - n_mels : mel scale을 만들기 위해 적용하는 mel filter 의 개수

### librosa.power_to_db

만들어진 mel spectrogram이 power scale일 경우 변화를 log scale로 인식할 수 있도록 log 변환을 해주는 과정입니다.

### Data Augmentation : Random Eraser & Imagegenerator

이미지를 shift하고, random하게 이미지의 일부를 지우는 과정을 통해 데이터를 증강하였습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b83f2e00-8586-4a03-b8cd-8018de5a7a07/Untitled.png)

# 3. 학습 모델

모델은 1개의 input layer, 4개의 hidden layer, 1개의 out layer로 총 7개로 구성하였습니다. 

data가 numpy 행렬이므로 input의 channel은 1이고 크기는 64*501이기 때문에 input layer에서 input shape로 이 값을 설정해주었습니다. 

그리고 합성곱 연산을 한 뒤 활성화함수로는 ReLU를 사용하였고 batch normalization을 통해 weight를 설정해주었습니다. 이 과정을 반복한뒤 Average pooling을 적용하는 방식으로 hidden layer를 구성하였습니다.

output layer는 마지막으로 output을 출력하는 층에서는 softmax 함수를 사용하여 각각의 class에 속할 확률을 나타내주고고 class가 6개이기 때문에 unit을 6으로 설정하였습니다.

아래 사진은 SVG 라이브러리를 이용하여 모델구조를 시각화한 결과입니다.

![dotres (2).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c31b19ec-b47a-40aa-8210-49abba83cc44/dotres_(2).png)

- model.summary()

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bcb15cb1-501b-415e-9c8c-6f8ed110e67d/Untitled.png)
