# d-vector-language-recognition
d-vector based language identification using pytorch


## Training
This steps starts training model 
```
python training.py --training_filepath meta/training.txt --testing_filepath meta/testing.txt --testing_filepath meta/validation_filepath
                             --input_dim 257 --num_classes 8 --lamda_val 0.1
                             --batch_size 10 --use_gpu True --num_epochs 100
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)



## References
##### 1. GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION.[\[Paper\]](https://arxiv.org/pdf/1710.10467.pdf)
##### 2. RawNet: Advanced end-to-end deep neural network using raw waveformsfor text-independent speaker verification.[\[Paper\]](https://arxiv.org/pdf/1904.08104.pdf)
##### 3. Language Independent Gender Identification from Raw Waveform Using Multi-Scale Convolutional Neural Networks.[\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9054738)

 Credits: G2E loss funtion code is taken from https://github.com/HarryVolek/PyTorch_Speaker_Verification
 
